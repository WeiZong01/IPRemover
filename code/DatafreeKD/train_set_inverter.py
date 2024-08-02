import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from DatafreeKD.hooks import DeepInversionHook
from torchvision import transforms
from kornia import augmentation
from utils import utils
from pathlib import Path
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import filelock


class TrainSetInverter:
    def __init__(self, out_dir, victim, mean_val, std_val,
                 generator_creator, batch_size, gen_num_per_class, num_classes,
                 gen_iters,
                 data_transforms=None):
        super().__init__()
        self.out_dir = out_dir
        self.victim = victim

        self.generator_creator = generator_creator
        self.batch_size = batch_size
        self.gen_num_per_class = gen_num_per_class
        self.num_classes = num_classes
        self.gen_iters = gen_iters

        self.hooks = []
        for m in victim.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )
        assert len(self.hooks) > 0, "No BatchNorm layers found in the target model."

        self.normalizer = transforms.Normalize(mean=mean_val, std=std_val)

        if data_transforms is not None:
            self.transforms = data_transforms
        else:
            self.transforms = transforms.Compose([
                augmentation.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.Normalize(mean=mean_val, std=std_val),
            ])

        self.alpha_bn_base = 1.0
        self.alpha_bn_multi = 2.0

    @staticmethod
    def get_batch_fname(batch_idx):
        return f"generated_batch_{batch_idx}.npz"

    def get_next_batch_path(self, batch_dir):
        gen_cnt_dic = {_class: 0 for _class in range(self.num_classes)}  # count how many data have generated for each class

        batch_idx = 0
        while True:
            batch_path = batch_dir.joinpath(self.get_batch_fname(batch_idx))
            batch_idx += 1

            lock_path = batch_path.with_suffix(".lock")
            lock = filelock.FileLock(lock_path, timeout=0)
            try:
                lock.acquire(timeout=0.5)     # give a short time to timeout. Other processes may also be visiting this file

                # if the batch file exists, then accumulate data in this file
                if batch_path.exists():
                    lock.release()      # do not need this lock anymore

                    gen_saved = np.load(batch_path)
                    gen_x, gen_y = gen_saved["x"], gen_saved["y"]

                    # count the generated number
                    for _y in gen_y:
                        gen_cnt_dic[_y] += 1

                else:
                    # the file does not exist, check if we need to generate more
                    cnt_lst = [_cnt for _, _cnt in gen_cnt_dic.items()]

                    already_gen = sum(cnt_lst)
                    total_to_gen = self.num_classes * self.gen_num_per_class
                    print(f"Total Generated {already_gen} (/ {total_to_gen}): {gen_cnt_dic}")

                    if min(cnt_lst) >= self.gen_num_per_class:
                        print("all data have been generated")
                        lock.release()
                        return None

                    # we need to generate data for this batch file
                    return batch_path, lock, gen_cnt_dic

            except filelock.Timeout as e:
                print(f"!!! {batch_path.name} is locked, try next one !!!")

    def synthesize(self):
        batch_dir = self.out_dir.joinpath("generated_batches")
        if not batch_dir.exists():
            Path.mkdir(batch_dir, parents=True)

        while True:

            next_batch = self.get_next_batch_path(batch_dir)
            if next_batch is None:
                # all generated. no lock to release either.
                return

            batch_path, lock, gen_cnt_dic = next_batch
            assert not batch_path.exists(), "The new batch file to generate must not exist."

            print(f"Generating {batch_path.name} ......")
            gen_x, gen_y = self.do_generate(gen_cnt_dic)
            gen_x = gen_x.cpu().detach().numpy()
            gen_y = gen_y.cpu().detach().numpy()

            valid_x, valid_y = [], []
            for _x, _y in zip(gen_x, gen_y):
                if gen_cnt_dic[_y] >= self.gen_num_per_class:
                    # data for this class are sufficient
                    continue
                valid_x.append(np.expand_dims(_x, axis=0))
                valid_y.append(_y)
                gen_cnt_dic[_y] += 1

            if len(valid_y) == 0:
                print("Failed to generate useful data.")
            else:
                # save generated data
                valid_x = np.concatenate(valid_x, axis=0)
                valid_y = np.array(valid_y)

                np.savez(batch_path, x=valid_x, y=valid_y)
                utils.save_image_batch(valid_x, valid_y, num_classes=self.num_classes, output=batch_path.with_name(batch_path.stem+".png"))

            lock.release()  # !!! remember to release the lock

        # finish

    def do_generate(self, gen_cnt_dic):
        self.victim.eval()  # the victim is not modified

        # create a new generator each time
        generator = self.generator_creator().to(utils.device)
        generator.train()

        z = torch.randn(size=(self.batch_size, generator.z_dim), device=utils.device)
        optimizer = torch.optim.Adam([{'params': generator.parameters()}], 1e-3)

        # targets must contain labels that are not fully generated yet
        labels_lacking = []
        for label, cnt in gen_cnt_dic.items():
            if cnt < self.gen_num_per_class:
                labels_lacking.append(label)
        assert len(labels_lacking) > 0, "Nothing to generate, something wrong!"

        while True:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.batch_size,))

            has_lack = False
            for label in labels_lacking:
                if (targets == label).sum().item() > 0:
                    has_lack = True
                    break
            if has_lack is True:
                break

        targets = targets.to(utils.device)

        gen_batch = None  # generated batch
        pbar = tqdm(range(self.gen_iters))

        # decrease learning rate after 75% progress
        scheduler = MultiStepLR(optimizer, milestones=[int(self.gen_iters * 0.75)], gamma=0.1)

        alpha_bn = torch.rand(len(self.hooks)).to(utils.device) * self.alpha_bn_multi + self.alpha_bn_base

        for it in pbar:
            gen_batch = generator(z)
            gen_transformed = self.transforms(gen_batch)

            #############################################
            # Inversion Loss
            #############################################
            t_out = self.victim(gen_transformed)

            loss_bn = torch.concat([h.r_feature.reshape([1]) for h in self.hooks])

            ce_loss = F.cross_entropy(t_out, targets)

            loss_inv = (alpha_bn * loss_bn).sum() + ce_loss

            optimizer.zero_grad()
            loss_inv.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"training: iter {it + 1} / {self.gen_iters}; "
                                 f"lr = {scheduler.get_last_lr()[0]:e}; "
                                 f"loss {loss_inv:.3f};")

        # only return correctly recognized ones
        preds = self.victim(self.normalizer(gen_batch))
        preds = torch.argmax(preds, dim=1)
        correct = (preds == targets)

        return gen_batch[correct], targets[correct]





