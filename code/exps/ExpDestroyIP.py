import copy
import pickle
from collections import Counter
from datetime import datetime

import torchvision.models
from tqdm import tqdm
from models import ModelTrainer
from exps.ExpBase import ExpBase
from utils import utils
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from DatafreeKD.train_set_inverter import TrainSetInverter
import logging


class ExpDestroyIP(ExpBase):
    """

    """
    def __init__(self, out_dir, victim_model, victim_creator,
                 inv_gen_creator, inv_batch_size, inv_gen_num_per_class,
                 inv_gen_iters, inv_max_filter_num,
                 in_distri_set, unlabelled_set, test_set,
                 train_cfg, distill_cfg,
                 dset_mean, dset_std, dset_preprocess,

                 do_ensemble_distill=True, self_distill_prob=0.8, ensemble_alpha=[0.3, 0.7], patience=0,

                 remove_gen_data=False, merge_in_distri_set=False, in_distri_set_labelled=True,
                 merge_unlabelled_set=False, upsample_factor=1.0, merge_prob_threshold=0.0,

                 train_epoch_callbk=None, distill_epoch_callbk=None, gen_data_upsample_factor=None,

                 train_ckpt_dname="train_ckpt", distill_ckpt_dname="distill_ckpt",

                 exp_status_fname="exp.status"):
        """
        gen_data_upsample_factor: only used for upsampling generated data
        """
        super().__init__(out_dir, exp_status_fname)

        self.victim_model = victim_model
        self.victim_creator = victim_creator

        self.inv_gen_creator = inv_gen_creator
        self.inv_batch_size = inv_batch_size
        self.inv_gen_num_per_class = inv_gen_num_per_class

        self.inv_gen_iters = inv_gen_iters
        self.inv_max_filter_num = inv_max_filter_num

        self.in_distri_set = in_distri_set
        self.unlabelled_set = unlabelled_set
        self.test_set = test_set

        self.train_cfg = copy.copy(train_cfg)
        if self.train_cfg is not None:
            self.train_cfg.ckpt_dir = self.out_dir.joinpath(train_ckpt_dname)
            logging.info(f"changing self.train_cfg.ckpt_dir -->> {self.train_cfg.ckpt_dir}")

        self.distill_cfg = copy.copy(distill_cfg)
        if self.distill_cfg is not None:
            self.distill_cfg.ckpt_dir = self.out_dir.joinpath(distill_ckpt_dname)
            logging.info(f"changing self.distill_cfg.ckpt_dir -->> {self.distill_cfg.ckpt_dir}")

        self.do_ensemble_distill = do_ensemble_distill

        self.dset_mean = dset_mean
        self.dset_std = dset_std

        self.dset_preprocess = dset_preprocess

        self.train_epoch_callbk = train_epoch_callbk
        self.distill_epoch_callbk = distill_epoch_callbk

        self.merge_prob_threshold = merge_prob_threshold

        self.self_distill_prob = self_distill_prob

        self.num_classes = len(self.test_set.targets.unique())

        self.ensemble_alpha = ensemble_alpha

        self.remove_gen_data = remove_gen_data
        self.merge_in_distri_set = merge_in_distri_set
        self.in_distri_set_labelled = in_distri_set_labelled
        self.merge_unlabelled_set = merge_unlabelled_set
        self.upsample_factor = upsample_factor

        self.gen_data_upsample_factor = gen_data_upsample_factor
        if self.gen_data_upsample_factor is None:
            if self.merge_unlabelled_set is False and self.merge_in_distri_set is False:
                logging.info(f"WARNING!!! gen_data_upsample_factor = None while no extra data to merge. "
                             f"It is better to set gen_data_upsample_factor=2 in this case "
                             f"because the number of epochs are set based on doubling the training set.")

        # early stopping
        self.best_train_acc = 0.0
        self.patience = patience
        self.tolerance_times = 0
        self.train_acc_delta = -100     #    0.001    # improvement (set to -100 means no early stopping)
        self.best_epoch = 0
        self.best_model_state = None

    def save_status(self):
        super().save_status()

    def teacher_logits_modifier(self, trainer, data, logits, target, cur_epoch, it, teacher_logits, extra_data):
        if extra_data is None:
            # do not consider agreement between student and teacher
            return teacher_logits

        #     assert False
        #     assert self.high_conf_prob > 0.5, "this probability must dominate the prediction."
        #     teacher_probs = torch.softmax(teacher_logits, dim=1)
        #     target_probs = torch.gather(teacher_probs, 1, target.unsqueeze(-1))
        #     to_modify = target_probs.squeeze() < self.high_conf_prob

        # do teacher free KD
        stu_preds = torch.argmax(logits, dim=1)
        to_modify = (stu_preds == stu_preds)    # modify all

        if torch.any(to_modify).item():
            # calculate a reverse solution for softmax
            prob = extra_data  # probability output by the teacher
            modified_logit_val = np.log(prob)
            other_val = np.log((1 - prob) / (self.num_classes - 1))

            # change logits to desired values
            target_to_modify = stu_preds[to_modify]

            teacher_logits[to_modify] = other_val
            teacher_logits[to_modify, target_to_modify] = modified_logit_val

        return teacher_logits

    def my_distill_epoch_callbk(self, epoch, train_loss, train_acc, test_loss, test_acc, optimizer, scheduler, trainer):
        if self.distill_epoch_callbk is not None:
            self.distill_epoch_callbk(epoch, train_loss, train_acc, test_loss, test_acc, optimizer, scheduler, trainer)

        # may need to change the learning procedure earlier than expected
        assert len(scheduler.milestones) == 1, "only support one step_size now"
        milestone = list(scheduler.milestones.keys())[0]

        if epoch >= milestone:
            # the milestone has already passed, do nothing
            return True

        if train_acc > self.best_train_acc + self.train_acc_delta:
            self.best_train_acc = train_acc
            self.tolerance_times = 0

            # save the model state for recovering
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(trainer.model.state_dict())

        else:
            self.tolerance_times += 1
            logging.info(f"tolerance_times = {self.tolerance_times}.")

            if self.tolerance_times > self.patience:
                logging.info(f"tolerance_times > self.patience: entering the mile stone earlier "
                             f"and recover to epoch = {self.best_epoch}")
                trainer.model.load_state_dict(self.best_model_state)

                # use scheduler to change the learning rate
                scheduler.milestones = Counter([epoch+1])
                scheduler.step()            # this will make the step of scheduler and the optimizer inconsistent, but we don't care.
                scheduler.milestones = Counter([epoch])

                trainer.config.max_epochs = epoch + (trainer.config.max_epochs - milestone)

        return True

    def get_inv_dir_name(self):
        return f"train_set_inv_iter_{self.inv_gen_iters}_inv_filter_{self.inv_max_filter_num}_batch_{self.inv_batch_size}"

    def create_inverter(self):
        return TrainSetInverter(
            self.out_dir.joinpath(self.get_inv_dir_name()),
            victim=self.victim_model,
            mean_val=self.dset_mean, std_val=self.dset_std,
            generator_creator=self.inv_gen_creator, batch_size=self.inv_batch_size,
            gen_num_per_class=self.inv_gen_num_per_class,
            num_classes=self.test_set.targets.unique().shape[0],
            gen_iters=self.inv_gen_iters,
        )

    def generate_train_set(self):
        inverter = self.create_inverter()
        inverter.synthesize()

    def train_base_model(self, gen_x, gen_y, remove_gen_data, merge_in_distri_set, in_distri_set_labelled,
                         merge_unlabelled_set, upsample_factor):
        # first generate training data from the victim model
        train_set = utils.MyNumpyData(gen_x, gen_y, preprocess=self.dset_preprocess)

        if upsample_factor is None:
            upsample_to = None
        else:
            upsample_to = int(len(train_set) * upsample_factor)

        if remove_gen_data is False:
            logging.info(f"Do not remove generated data.")

            logging.info(f"self.gen_data_upsample_factor = {self.gen_data_upsample_factor}")
            if self.gen_data_upsample_factor is not None:
                assert isinstance(self.gen_data_upsample_factor, int) and self.gen_data_upsample_factor > 1, "gen_data_upsample_factor must be an integer > 1"
                train_set.np_data = np.repeat(train_set.np_data, self.gen_data_upsample_factor, axis=0)
                train_set.np_labels = np.repeat(train_set.np_labels, self.gen_data_upsample_factor, axis=0)
                logging.info(f"Upsampled generated data to {len(train_set)}")

        else:
            logging.info(f"remove generated data, remove_gen_data = {remove_gen_data}")
            if remove_gen_data is True:
                train_set.np_data = None
                train_set.np_labels = None
            else:
                assert 0.0 < remove_gen_data < 1.0
                gen_keep = np.ceil((1 - remove_gen_data) * len(train_set) / self.num_classes)

                gen_cnt_dic = {_class: 0 for _class in range(self.num_classes)}
                keep_mask = [True] * len(train_set)
                for idx, label in enumerate(train_set.np_labels):
                    if gen_cnt_dic[label] < gen_keep:
                        gen_cnt_dic[label] += 1
                    else:
                        keep_mask[idx] = False

                train_set.np_data = train_set.np_data[keep_mask]
                train_set.np_labels = train_set.np_labels[keep_mask]

                logging.info(f"now len(train_set) = {len(train_set)}")

        if merge_in_distri_set:
            logging.info("merge in distribution set")
            self.merge_set(target_set=train_set, src_set=self.in_distri_set, src_labelled=in_distri_set_labelled,
                           src_perc_thres=self.merge_prob_threshold,
                           upsample_to=upsample_to, mean_val=self.dset_mean, std_val=self.dset_std)

        if merge_unlabelled_set:
            logging.info("merge unlabelled set")
            self.merge_set(target_set=train_set, src_set=self.unlabelled_set, src_labelled=False,
                           src_perc_thres=self.merge_prob_threshold,
                           upsample_to=upsample_to, mean_val=self.dset_mean, std_val=self.dset_std)

        base_model = self.victim_creator()
        base_model = base_model.to(utils.device)

        trainer = ModelTrainer.ModelTrainer(base_model, train_set, self.test_set, self.train_cfg)

        self.train_cfg.train_epoch_callback = self.train_epoch_callbk
        self.train_cfg.train_epoch_callback_data = trainer

        trainer.run()

        return train_set, base_model

    def distill(self, train_set, extra_teacher):

        # load pretrained model if exists
        stu_model = self.victim_creator()

        trained_dic = ModelTrainer.ModelTrainer.load_latest_ckpt(self.train_cfg.ckpt_dir)
        if trained_dic is not None:
            stu_model.load_state_dict(trained_dic["model_state"])
            logging.info("Loaded trained student model for distillation. Evaluating the loaded model...")
            ModelTrainer.ModelTrainer.eval_on_dset(stu_model, self.test_set)
        else:
            logging.info("Did not find existing trained model for distillation")
            assert self.train_cfg.max_epochs == 0, "train_cfg.max_epochs must have been set to 0 in this case."

        distill_loss_cfg = self.distill_cfg.loss_other_data

        if extra_teacher is not None:
            logging.info(f"Distillation with an extra teacher. self.self_distill_prob = {self.self_distill_prob}; "
                         f"self.ensemble_alpha = {self.ensemble_alpha}")
            distill_loss_cfg.teacher_model = [self.victim_model, stu_model]
            distill_loss_cfg.ensemble_alpha = self.ensemble_alpha
            distill_loss_cfg.extra_data = [None, self.self_distill_prob]

        else:
            logging.info(f"Naive knowledge distillation with only the victim model.")
            # remember to set these values
            distill_loss_cfg.teacher_model = self.victim_model
            distill_loss_cfg.ensemble_alpha = 1.0
            distill_loss_cfg.extra_data = None

        distill_loss_cfg.teacher_logits_modifier = self.teacher_logits_modifier  # may need to change incorrect predictions

        trainer = ModelTrainer.ModelTrainer(stu_model, train_set, self.test_set, self.distill_cfg)

        self.distill_cfg.train_epoch_callback = self.my_distill_epoch_callbk
        self.distill_cfg.train_epoch_callback_data = trainer

        trainer.run()

    def merge_set(self, target_set, src_set, src_perc_thres, src_labelled, upsample_to, mean_val, std_val):
        src_x, src_y = [], []

        if src_labelled:
            # directly using the ground truth
            for x, y in src_set:
                assert x.max().item() <= 1.0 and x.min().item() >= 0.0, "extra data should not be normalized initially"

                if not isinstance(y, torch.Tensor):
                    y = torch.LongTensor([y]).squeeze()

                src_x.append(x.unsqueeze(0))
                src_y.append(y.reshape([1]))

        else:
            # use the victim to label src set
            d_loader = DataLoader(src_set, batch_size=1024, shuffle=False, num_workers=8)

            mean_val = torch.FloatTensor(mean_val).reshape([1, 3, 1, 1]).to(utils.device)
            std_val = torch.FloatTensor(std_val).reshape([1, 3, 1, 1]).to(utils.device)

            self.victim_model.eval()
            for x, y in tqdm(d_loader):
                x = x.to(utils.device)
                assert x.max().item() <= 1.0 and x.min().item() >= 0.0, "extra data should not be normalized initially"

                x_normalized = (x - mean_val) / std_val

                with torch.no_grad():
                    logits = self.victim_model(x_normalized)
                probs = torch.softmax(logits, dim=1)
                probs = torch.max(probs, dim=1)[0]
                to_keep = probs > src_perc_thres
                if torch.any(to_keep).item() is True:
                    src_x.append(x[to_keep])
                    unlabelled_pred = torch.argmax(logits, dim=1)
                    src_y.append(unlabelled_pred[to_keep])

        extra_data = torch.concat(src_x, axis=0).cpu().detach().numpy()
        extra_labels = torch.concat(src_y, axis=0).cpu().detach().numpy()

        extra_size = len(extra_data)

        # upsample the data if necessary
        if upsample_to is not None:
            upsample = np.ceil(upsample_to / len(extra_data))
            extra_data = np.repeat(extra_data, upsample, axis=0)
            extra_labels = np.repeat(extra_labels, upsample, axis=0)

            selected_idxes = np.random.RandomState(seed=32987).permutation(len(extra_data))
            selected_idxes = selected_idxes[:upsample_to]
            extra_data = extra_data[selected_idxes]
            extra_labels = extra_labels[selected_idxes]

        target_set.add_extra_data(extra_data, extra_labels)

        print(f"!!! Merged data = {extra_size}, src_labelled = {src_labelled}, up sampled to {upsample_to}, "              
              f"src_perc_thres = {src_perc_thres}, now total size = {len(target_set)} !!!")

    def combine_inverted_data(self, inv_dir=None, combined_path=None, remove_extra=True):
        if inv_dir is None:
            inv_dir = self.out_dir.joinpath(self.get_inv_dir_name())
        else:
            inv_dir = inv_dir.joinpath(self.get_inv_dir_name())

        if combined_path is None:
            combined_path = inv_dir.joinpath(f"generated_combined_{self.inv_gen_num_per_class}.npz")

        if combined_path.exists():
            combined_loaded = np.load(combined_path)
            combined_x, combined_y = combined_loaded["x"], combined_loaded["y"]

            total_gen_num = self.inv_gen_num_per_class * self.num_classes

            logging.info(f"Loaded existing combined data: combined_x = {len(combined_x)}, combined_y = {len(combined_y)}; "
                         f"remove_extra = {remove_extra}")
            if remove_extra is True:
                assert (len(combined_x) == total_gen_num) and (len(combined_y) == total_gen_num)
            else:
                assert (len(combined_x) >= total_gen_num) and (len(combined_y) >= total_gen_num)

            return combined_x, combined_y

        # combine all the batches into one large file for quick loading
        gen_x, gen_y = [], []
        batch_dir = inv_dir.joinpath("generated_batches")
        batch_idx = 0

        while True:
            batch_path = batch_dir.joinpath(TrainSetInverter.get_batch_fname(batch_idx))
            batch_idx += 1

            if not batch_path.exists():
                break

            data_loaded = np.load(batch_path)
            x, y = data_loaded["x"], data_loaded["y"]
            gen_x.append(x)
            gen_y.append(y)

        if len(gen_x) == 0:
            return None, None

        gen_x = np.concatenate(gen_x, axis=0)
        gen_y = np.concatenate(gen_y, axis=0)
        assert len(gen_x) == len(gen_y)

        # check whether the number of data is sufficient
        for label in range(self.num_classes):
            if (gen_y == label).sum() < self.inv_gen_num_per_class:
                return None, None

        if remove_extra is True:
            # remove extra data to make all data <= inv_gen_num_per_class
            gen_cnt_dic = {_class: 0 for _class in range(self.num_classes)}
            mask_keep = np.array([True]*len(gen_x))

            for idx, label in enumerate(gen_y):
                if gen_cnt_dic[label] < self.inv_gen_num_per_class:
                    gen_cnt_dic[label] += 1
                else:
                    mask_keep[idx] = False

            gen_x = gen_x[mask_keep]
            gen_y = gen_y[mask_keep]
            assert len(gen_x) == self.inv_gen_num_per_class * self.num_classes

        logging.info(f"Combined: gen_x = {len(gen_x)}; gen_y = {len(gen_y)}; remove_extra = {remove_extra}")

        np.savez(combined_path, x=gen_x, y=gen_y)

        return gen_x, gen_y

    def run(self, inv_dir=None, combined_path=None, remove_extra=True, flag_name=f"ExpDestroyIP_run_finished_flag.bin"):
        """
        To bypass IP protection, first train a model from scratch,
        then distill knowledge from the victim model
        """
        flag_path = self.out_dir.joinpath(flag_name)
        if flag_path.exists():
            with open(flag_path, 'rb') as handle:
                flag = pickle.load(handle)
            print(f"{self.out_dir.name} has already done at {flag}.")
            return

        # first get all generated dataset

        gen_x, gen_y = self.combine_inverted_data(inv_dir=inv_dir, combined_path=combined_path, remove_extra=remove_extra)
        if gen_x is None:
            assert False, "Generated data are not sufficient for training models. Make sure to run generation completely first."

        train_set, base_model = self.train_base_model(
            gen_x, gen_y,
            remove_gen_data=self.remove_gen_data,
            merge_in_distri_set=self.merge_in_distri_set, in_distri_set_labelled=self.in_distri_set_labelled,
            merge_unlabelled_set=self.merge_unlabelled_set, upsample_factor=self.upsample_factor,
        )

        if self.distill_cfg is not None:
            # do distill only when distill_cfg is set
            if self.do_ensemble_distill:
                self.distill(train_set, base_model)
            else:
                self.distill(train_set, None)

        self.save_status()

        flag = datetime.now()
        with open(flag_path, 'wb') as handle:
            pickle.dump(flag, handle)

    def inverting_or_running(self, flag_name=f"ExpDestroyIP_run_finished_flag.bin"):
        # check if we need to invert more data
        inverter = self.create_inverter()
        batch_dir = inverter.out_dir.joinpath("generated_batches")
        if not batch_dir.exists():
            Path.mkdir(batch_dir, parents=True)
        next_batch = inverter.get_next_batch_path(batch_dir)
        if next_batch is None:
            # all generated
            inverting = False
            self.run(flag_name=flag_name)
        else:
            batch_path, lock, gen_cnt_dic = next_batch
            lock.release()  # !!! remember to release the lock anyway

            # still need more data
            inverting = True
            self.generate_train_set()

        return inverting





