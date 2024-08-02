import sys
import argparse
import torch
from torchvision import transforms
from pathlib import Path

from models.wide_resnet import cifar_wide_resnet
from models import ModelTrainer
from exps.ExpDestroyIP import ExpDestroyIP
from DatafreeKD.my_generator import Generator
from utils import utils

import logging
logging.basicConfig(level=20)


class InputTransformModel(torch.nn.Module):
    def __init__(self, model,  normalize=None, input_trans=None):
        super(InputTransformModel, self).__init__()

        self.model = model
        self.input_trans = input_trans
        self.normalize = normalize

    def __normalize(self, x):
        mean = torch.tensor(self.normalize[0], dtype=x.dtype, device=x.device)
        std = torch.tensor(self.normalize[1], dtype=x.dtype, device=x.device)
        x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
        return x

    def forward(self, x):

        if self.input_trans:
            x = self.input_trans(x)

        if self.normalize:
            x = self.__normalize(x)

        return self.model(x)

    def penultimate_forward(self, x):

        if self.input_trans:
            x = self.input_trans(x)

        if self.normalize:
            x = self.__normalize(x)

        return self.model.penultimate_forward(x)


def load_model(model_creator, ckpt_path):
    if ckpt_path.suffix == "":
        # this is directory, read using our ModelTrainer
        dic_saved = ModelTrainer.ModelTrainer.load_latest_ckpt(ckpt_path)
        dic_saved = dic_saved["model_state"]
    else:
        dic_saved = torch.load(ckpt_path)
        if "model" in dic_saved:
            dic_saved = dic_saved["model"]
        elif "model_state" in dic_saved:
            dic_saved = dic_saved["model_state"]
        else:
            pass    # do nothing

    model = model_creator()
    model.load_state_dict(dic_saved)
    model = model.to(utils.device)

    return model


def eval_query_set(model, dset_mean, dset_std, queryset, querylabels):
    # evaluate the stolen model
    model.eval()
    with torch.no_grad():
        model = InputTransformModel(model, normalize=(dset_mean, dset_std))
        model = model.to(utils.device)
        model.eval()
        preds = model(queryset)
        preds = preds.argmax(1).cpu().numpy()

    query_acc = (preds == querylabels).sum() / querylabels.size
    return query_acc


def eval_model(model, test_set):

    print("evaluating accuracy on the original testing set.")
    ModelTrainer.ModelTrainer.eval_on_dset(model, test_set)

    # evaluate the victim on fingerprints. We have provided our generated fingerprints together with this code
    queryset = torch.load(exp_cfg.mf_query_path).to(utils.device)
    querylabels = torch.load(exp_cfg.mf_label_path).numpy()

    victim_query_acc = eval_query_set(model, utils.CIFAR10_MEAN, utils.CIFAR10_STD, queryset, querylabels)
    print(f"accuracy on MetaFinger query set: {victim_query_acc}")


def main():

    out_dir = Path(exp_cfg.out_dir)

    # download the testing set for testing the victim model and stolen model
    test_set = utils.MyCIFAR10(out_dir, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(utils.CIFAR10_MEAN, utils.CIFAR10_STD),
    ]))

    # evaluate the victim
    victim_model = load_model(cifar_wide_resnet, Path(exp_cfg.victim_ckpt))
    print("***** Evaluating victim model: *****")
    eval_model(victim_model, test_set)

    exp = ExpDestroyIP(
        out_dir=out_dir,
        victim_model=victim_model, victim_creator=cifar_wide_resnet,

        inv_gen_creator=lambda: Generator(max_filter_num=exp_cfg.inv_max_filter_num, img_size=32, img_channel=3),
        inv_batch_size=exp_cfg.inv_batch_size, inv_gen_num_per_class=5000,
        inv_gen_iters=exp_cfg.inv_gen_iters, inv_max_filter_num=exp_cfg.inv_max_filter_num,

        in_distri_set=None, unlabelled_set=None, test_set=test_set,

        # gen_data_upsample_factor=2 means we double the generated data via upsampling
        # This makes computations consistent with the case when labeled training data are upsampled and mixed
        gen_data_upsample_factor=2,

        # 15 epochs mean the real epochs are equal to 30 because the generated data are doubled by upsampling.
        train_cfg=ModelTrainer.ModelTrainerConfig(
            batch_size=128,
            num_workers=16,
            max_epochs=15,
            lr=1e-3,
            lr_gamma=1.0,       # do not decrease lr
        ),

        # 2 epochs mean the real epochs are equal to 4 because the generated data are doubled by upsampling.
        distill_cfg=ModelTrainer.ModelTrainerConfig(
            batch_size=128,
            num_workers=16,
            max_epochs=2,
            lr=1e-3,
            lr_gamma=0.1,
            lr_step_size=[1],

            loss_func=ModelTrainer.distill_loss,
            loss_other_data=ModelTrainer.DistillLossCfg(),
        ),

        dset_mean=utils.CIFAR10_MEAN, dset_std=utils.CIFAR10_STD,
        dset_preprocess=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.9, 1.1)),
            transforms.Normalize(utils.CIFAR10_MEAN, utils.CIFAR10_STD),
        ]),

    )

    # Generating 5000 images for each class may take over 10 hours.
    # You may run this code in multiple processes to speed up generation.
    # Our generated data are too large to be uploaded during this review stage.

    # If more data need to be generated, inverting_or_running will only generate data.
    # If all data are generated, inverting_or_running will train a stolen model instead.
    inverting = exp.inverting_or_running()
    if inverting is True:
        # inverting data now. A stolen model has not been generated
        return

    # evaluate the stolen model
    stolen_ckpt_dir = out_dir.joinpath("distill_ckpt")
    stolen_model = load_model(cifar_wide_resnet, stolen_ckpt_dir)
    print("***** Evaluating stolen model: *****")
    eval_model(stolen_model, test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # model inversion
    parser.add_argument('--inv_max_filter_num', default=256, type=int,
                        help='number of maximum filters for the generator')
    parser.add_argument('--inv_batch_size', default=64, type=int,
                        help='batch size for inverting model')
    parser.add_argument('--inv_gen_iters', default=2000, type=int,
                        help='number of iterations to run')

    # paths
    parser.add_argument('--out_dir', default="../", type=str,
                        help='directory for output')

    parser.add_argument('--victim_ckpt', default="../source_model.pth", type=str,
                        help='This is a pretrained model published by MetaFinger: '
                             'cifar10/train_data/positive_models/source_model.pth')

    parser.add_argument('--mf_query_path', default="../fingerprints/cifar10_meta_queryset.pth", type=str,
                        help='Our generated Metafinger query data')
    parser.add_argument('--mf_label_path', default="../fingerprints/cifar10_meta_querylabels.pth", type=str,
                        help='Our generated Metafinger query labels')

    exp_cfg = parser.parse_args()

    main()

    sys.exit()





