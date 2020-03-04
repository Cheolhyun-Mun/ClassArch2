from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import pytorch_utils as pt_utils
from utils import viz as v
import pprint
import os.path as osp
import argparse

from models.rscnn_ssn_cls import RSCNN_SSN as RSCNN
from models.rscnn_ssn_cls import model_fn_decorator
from data.ModelNet40Loader import ModelNet40
import data.data_utils as d_utils
# from RandAugment3D.augmentation import RandAugment3D

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=1024, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=21, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.9, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=400, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom", action="store_true")

    return parser.parse_args()


lr_clip = 0.00001
bnm_clip = 0.01

if __name__ == "__main__":
    args = parse_args()

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
        ]
    )

    test_set = ModelNet40(args.num_points, transforms=transforms, split='test')
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    train_set = ModelNet40(args.num_points, transforms=transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = RSCNN(input_channels=0, num_classes=40, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda it: max(
        args.lr_decay ** (int(it // args.decay_step)),
        lr_clip / args.lr,
    )
    bn_lbmd = lambda it: max(
        args.bn_momentum
        * args.bnm_decay ** (int(it // args.decay_step)),
        bnm_clip,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    if args.visdom:
        viz = v.VisdomViz(port=args.visdom_port)
    else:
        viz = v.CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="checkpoints/rscnn_cls",
        best_name="checkpoints/rscnn_cls_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        viz=viz,
    )

    trainer.train(
        it, start_epoch, args.epochs, train_loader, test_loader, best_loss=best_loss
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(test_loader)
