#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import shutil
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from fvcore.nn import FlopCountAnalysis
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from unetr_pp.evaluation.evaluator import NiftiEvaluator, run_evaluation
from unetr_pp.inference.segmentation_export import save_segmentation_nifti_from_softmax
from unetr_pp.network_architecture.initialization import InitWeights_He
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from unetr_pp.postprocessing.connected_components import determine_postprocessing
from unetr_pp.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from unetr_pp.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from unetr_pp.training.dataloading.dataset_loading import unpack_dataset
from unetr_pp.training.learning_rate.poly_lr import poly_lr
from unetr_pp.training.loss_functions.deep_supervision import MultipleOutputLoss2
from unetr_pp.training.network_training.Trainer_acdc import Trainer_acdc
from unetr_pp.utilities.nd_softmax import softmax_helper
from unetr_pp.utilities.to_torch import maybe_to_torch, to_cuda


class unetr_pp_trainer_amos_mri(Trainer_acdc):
    """
    AMOS MRI trainer with a fixed 32/8 split:
    - first 32 sorted cases for training
    - next 8 sorted cases for validation
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False

        self.load_plans_file()

        self.crop_size = np.array([64, 128, 128])
        self.input_channels = self.plans['num_modalities']
        self.num_classes = self.plans['num_classes'] + 1
        self.conv_op = nn.Conv3d
        self.deep_supervision = True

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            # Keep architecture behavior close to Synapse setting for abdominal organs.
            self.plans['plans_per_stage'][self.stage]['patch_size'] = self.crop_size
            self.plans['plans_per_stage'][self.stage]['batch_size'] = 2
            self.plans['plans_per_stage'][self.stage]['pool_op_kernel_sizes'] = [[2, 4, 4], [2, 2, 2], [2, 2, 2]]
            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.deep_supervision:
                net_numpool = len(self.net_num_pool_op_kernel_sizes)
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            self.folder_with_preprocessed_data = join(self.dataset_directory,
                                                      self.plans['data_identifier'] + "_stage%d" % self.stage)
            seeds_train = np.random.randint(0, 100000, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.randint(0, 100000, max(self.data_aug_params.get('num_threads') // 2, 1))
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        self.network = UNETR_PP(
            in_channels=self.input_channels,
            out_channels=self.num_classes,
            img_size=self.crop_size,
            feature_size=16,
            num_heads=4,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
            do_ds=True,
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        input_res = (1, int(self.crop_size[0]), int(self.crop_size[1]), int(self.crop_size[2]))
        sample = torch.ones(()).new_empty((1, *input_res), dtype=next(self.network.parameters()).dtype,
                                          device=next(self.network.parameters()).device)
        flops = FlopCountAnalysis(self.network, sample)
        model_flops = flops.total()
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """Fully sequential validation — avoids multiprocessing Pool deadlocks
        that occur when spawn/fork workers interact with the CUDA runtime."""
        ds = self.network.do_ds
        self.network.do_ds = False
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(
                    data[:-1],
                    do_mirroring=do_mirroring,
                    mirror_axes=mirror_axes,
                    use_sliding_window=use_sliding_window,
                    step_size=step_size,
                    use_gaussian=use_gaussian,
                    all_in_gpu=all_in_gpu,
                    mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                # Export nifti directly in main process — no subprocess Pool.
                save_segmentation_nifti_from_softmax(
                    softmax_pred, join(output_folder, fname + ".nii.gz"),
                    properties, interpolation_order, self.regions_class_order,
                    None, None,
                    softmax_fname, None, force_separate_z,
                    interpolation_order_z)

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        self.print_to_log_file("finished prediction")

        # Evaluate sequentially — avoid Pool inside aggregate_scores.
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name

        evaluator = NiftiEvaluator()
        evaluator.set_labels(list(range(self.num_classes)))
        all_scores = OrderedDict()
        all_scores["all"] = []
        all_scores["mean"] = OrderedDict()

        for test_file, ref_file in pred_gt_tuples:
            scores = run_evaluation((test_file, ref_file, evaluator, {}))
            all_scores["all"].append(scores)
            for label, score_dict in scores.items():
                if label in ("test", "reference"):
                    continue
                if label not in all_scores["mean"]:
                    all_scores["mean"][label] = OrderedDict()
                for score, value in score_dict.items():
                    if score not in all_scores["mean"][label]:
                        all_scores["mean"][label][score] = []
                    all_scores["mean"][label][score].append(value)

        for label in all_scores["mean"]:
            for score in all_scores["mean"][label]:
                all_scores["mean"][label][score] = float(np.nanmean(all_scores["mean"][label][score]))

        json_dict = OrderedDict()
        json_dict["name"] = job_name + " val tiled %s" % (str(use_sliding_window))
        json_dict["author"] = "Fabian"
        json_dict["task"] = task
        json_dict["results"] = all_scores
        save_json(json_dict, join(output_folder, "summary.json"))

        if run_postprocessing_on_folds:
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)

        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    attempts += 1

        self.network.train(current_mode)
        self.network.do_ds = ds

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(
            data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            all_in_gpu=all_in_gpu,
            verbose=verbose,
            mixed_precision=mixed_precision
        )
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = maybe_to_torch(data_dict['data'])
        target = maybe_to_torch(data_dict['target'])

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                loss = self.loss(output, target)
            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            loss = self.loss(output, target)
            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return loss.detach().cpu().numpy()

    def do_split(self):
        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            if len(all_keys_sorted) >= 40:
                tr_keys = np.array(all_keys_sorted[:32])
                val_keys = np.array(all_keys_sorted[32:40])
                self.print_to_log_file("Using fixed AMOS MRI split: first 32 train, next 8 val.")
            else:
                self.print_to_log_file(
                    "Dataset has <40 cases. Falling back to seeded 80:20 split.")
                splits_file = join(self.dataset_directory, "splits_final.pkl")
                if not isfile(splits_file):
                    splits = []
                    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                    for train_idx, test_idx in kfold.split(all_keys_sorted):
                        train_keys = np.array(all_keys_sorted)[train_idx]
                        test_keys = np.array(all_keys_sorted)[test_idx]
                        splits.append(OrderedDict())
                        splits[-1]['train'] = train_keys
                        splits[-1]['val'] = test_keys
                    save_pickle(splits, splits_file)
                splits = load_pickle(splits_file)
                if self.fold < len(splits):
                    tr_keys = splits[self.fold]['train']
                    val_keys = splits[self.fold]['val']
                else:
                    rnd = np.random.RandomState(seed=12345 + self.fold)
                    idx_tr = rnd.choice(len(all_keys_sorted), int(len(all_keys_sorted) * 0.8), replace=False)
                    idx_val = [i for i in range(len(all_keys_sorted)) if i not in idx_tr]
                    tr_keys = [all_keys_sorted[i] for i in idx_tr]
                    val_keys = [all_keys_sorted[i] for i in idx_val]

        tr_keys = np.array(sorted(tr_keys))
        val_keys = np.array(sorted(val_keys))
        self.print_to_log_file("This split has %d training and %d validation cases."
                               % (len(tr_keys), len(val_keys)))

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform
        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs
        if self.epoch == 100 and self.all_val_eval_metrics[-1] == 0:
            self.optimizer.param_groups[0]["momentum"] = 0.95
            self.network.apply(InitWeights_He(1e-2))
            self.print_to_log_file("At epoch 100, mean foreground Dice was 0. Reduced momentum to 0.95 and reinit.")
        return continue_training

    def run_training(self):
        self.maybe_update_lr(self.epoch)
        ds = self.network.do_ds
        self.network.do_ds = bool(self.deep_supervision)
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
