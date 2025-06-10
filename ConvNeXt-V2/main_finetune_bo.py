# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine_finetune import train_one_epoch, evaluate

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, remap_checkpoint_keys
import models.convnextv2_unet as convnextv2_unet
import sys
from sklearn.model_selection import KFold

sys.path.append("..")
from gwfss_dataset import GWFSSDataset
from unbalanced_loss import get_class_balanced_weights

### For Bayesian optimization
from skopt.utils import use_named_args
from skopt.space import Integer, Categorical, Real
from skopt import gp_minimize

class SuppressPrint:
    def __enter__(self):
        # pass
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pass
        sys.stdout.close()
        sys.stdout = self._original_stdout  

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnextv2_unet_atto', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""")
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.3,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-2, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')    
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                       help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=4, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=True,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use apex AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--distributed', type=str2bool, default=False)

    parser.add_argument('--class_weights_beta', type=float, default=0.3)
    parser.add_argument('--cutoff_epoch', type=int, default=0)
    return parser

def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = GWFSSDataset(root_dir="/lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/data")


    
    ########    k fold cross validation    ########
    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = []
    with SuppressPrint():
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_train)):
            print(f"Fold {fold+1}/{k_folds}")

            train_subset = torch.utils.data.Subset(dataset_train, train_idx)
            val_subset = torch.utils.data.Subset(dataset_train, val_idx)
        
            data_loader_train = torch.utils.data.DataLoader(
                train_subset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
            data_loader_val = torch.utils.data.DataLoader(
                val_subset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            mixup_fn = None
            
            with SuppressPrint():
                model = convnextv2_unet.__dict__[args.model](
                    num_classes=args.nb_classes,
                    drop_path_rate=args.drop_path,
                    head_init_scale=args.head_init_scale,
                )
                
                if args.finetune:
                    checkpoint = torch.load(args.finetune, map_location='cpu')

                    print("Load pre-trained checkpoint from: %s" % args.finetune)
                    checkpoint_model = checkpoint['model']
                    state_dict = model.state_dict()
                    for k in ['head.weight', 'head.bias']:
                        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                            print(f"Removing key {k} from pretrained checkpoint")
                            del checkpoint_model[k]
                    

                    # remove decoder weights
                    checkpoint_model_keys = list(checkpoint_model.keys())
                    for k in checkpoint_model_keys:
                        if 'decoder' in k or 'mask_token'in k or \
                        'proj' in k or 'pred' in k:
                            print(f"Removing key {k} from pretrained checkpoint")
                            del checkpoint_model[k]
                    
                    checkpoint_model = remap_checkpoint_keys(checkpoint_model)
                    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
                    
                    # manually initialize fc layer
                    trunc_normal_(model.head.weight, std=2e-5)
                    torch.nn.init.constant_(model.head.bias, 0.)

                if args.cutoff_epoch > 0:
                    print("---unfreezing the decoder part of the model for unet---")
                    # we need to freeze the encoder part of the model
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    model.head.weight.requires_grad = True
                    model.head.bias.requires_grad = True

                    # we also have a list of upsample layers with upsample blocks in the model
                    for layer in model.upsample_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                    
                    for param in model.initial_conv_upsample.parameters():
                        param.requires_grad = True



                model.to(device)

                model_ema = None
                
                model_without_ddp = model
                n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

                print("Model = %s" % str(model_without_ddp))
                print('number of params:', n_parameters)

                eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
                num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
                
                if args.lr is None:
                    args.lr = args.blr * eff_batch_size / 256

                print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
                print("actual lr: %.2e" % args.lr)

                print("accumulate grad iterations: %d" % args.update_freq)
                print("effective batch size: %d" % eff_batch_size)

                if args.layer_decay < 1.0 or args.layer_decay > 1.0:
                    assert args.layer_decay_type in ['single', 'group']
                    if args.layer_decay_type == 'group': # applies for Base and Large models
                        num_layers = 12
                    else:
                        num_layers = sum(model_without_ddp.depths)
                    assigner = LayerDecayValueAssigner(
                        list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
                        depths=model_without_ddp.depths, layer_decay_type=args.layer_decay_type)
                else:
                    assigner = None

                if assigner is not None:
                    print("Assigned values = %s" % str(assigner.values))
                
            

                optimizer = create_optimizer(
                    args, model_without_ddp, skip_list=None,
                    get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                    get_layer_scale=assigner.get_scale if assigner is not None else None)
                loss_scaler = NativeScaler()

                class_counts = [8570875, 2768275, 905376, 13707730]
                class_weights = get_class_balanced_weights(class_counts, args.class_weights_beta).to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                
                print("criterion = %s" % str(criterion))

                utils.auto_load_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

                if args.eval:
                    print(f"Eval only mode")
                    test_stats = evaluate(data_loader_val, model, device)
                    print(f"Accuracy of the network on {len(val_subset)} test images: {test_stats['acc1']:.5f}%")
                    return
                
                max_accuracy = 0.0


            print("Start training for %d epochs" % args.epochs)
            start_time = time.time()
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)


                # for unet we probe the decoder only for 50 epochs, and then fine-tune for 150 epochs
                if epoch == args.cutoff_epoch and args.cutoff_epoch > 0:
                    print("Unfreezing the encoder part of the model")
                    for param in model.parameters():
                        param.requires_grad = True

                    new_param_groups = [
                        {"params": model.downsample_layers.parameters()},
                        {"params": model.stages.parameters()},
                    ]

                    optimizer.add_param_group({"params": [p for group in new_param_groups for p in group["params"]]})


                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler, 
                    args.clip_grad, model_ema, mixup_fn,
                    log_writer=None,
                    args=args
                )
                if args.output_dir and args.save_ckpt:
                    if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
                        
            # we only evaluate on the last epoch
            if data_loader_val is not None:
                test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
                mean_iou = (test_stats["mIoU_class_0"] + test_stats["mIoU_class_1"] + test_stats["mIoU_class_2"] + test_stats["mIoU_class_3"]) / 4
                print(f"Accuracy of the model on the {len(val_subset)} test images: {mean_iou:.1f}%")
            

            cv_scores.append({
                'fold': fold,
                'max_mean_iou': mean_iou,
                'mIoU_class_0': test_stats["mIoU_class_0"],
                'mIoU_class_1': test_stats["mIoU_class_1"],
                'mIoU_class_2': test_stats["mIoU_class_2"],
                'mIoU_class_3': test_stats["mIoU_class_3"],
            }) 



    for score in cv_scores:
        print(f'{"-"*100}')
        print(f"Fold {score['fold']+1} Mean IoU: {score['max_mean_iou']:.4f}%")
        print(f"Fold {score['fold']+1} mIoU Class 0: {score['mIoU_class_0']:.4f}%")
        print(f"Fold {score['fold']+1} mIoU Class 1: {score['mIoU_class_1']:.4f}%")
        print(f"Fold {score['fold']+1} mIoU Class 2: {score['mIoU_class_2']:.4f}%")
        print(f"Fold {score['fold']+1} mIoU Class 3: {score['mIoU_class_3']:.4f}%")

    print(f'{"-"*100}')
    print(f"Mean IoU: {np.mean([score['max_mean_iou'] for score in cv_scores]):.4f}%")
    print(f"Mean mIoU Class 0: {np.mean([score['mIoU_class_0'] for score in cv_scores]):.4f}%")
    print(f"Mean mIoU Class 1: {np.mean([score['mIoU_class_1'] for score in cv_scores]):.4f}%")
    print(f"Mean mIoU Class 2: {np.mean([score['mIoU_class_2'] for score in cv_scores]):.4f}%")
    print(f"Mean mIoU Class 3: {np.mean([score['mIoU_class_3'] for score in cv_scores]):.4f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    accuracy = np.mean([score['max_mean_iou'] for score in cv_scores])
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FCMAE fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Bayesian optimization
    n_calls = 200  # You can increase this for more thorough optimization
    # --- 2. Define the search space for hyperparameters ---
    space = [
        Integer(20, 200, name='epochs'),  # Number of training epochs
        Integer(8, 64, name='batch_size'),  # Batch size for training
        Real(0.0001, 0.01, name='blr'),
        # Real(0.99, 0.9999999, name='class_weights_beta'),
        Integer(0, 1, name='cutoff_epoch')
    ]

    @use_named_args(space)
    def objective(epochs, batch_size, blr, cutoff_epoch):

        print(f"\nSampled values: epochs = {epochs}, batch_size = {batch_size},"
              f"blr = {blr}, cutoff epoch = {cutoff_epoch}")

        epochs = int(epochs)
        batch_size = int(batch_size)
        cutoff_epoch = int(cutoff_epoch)

        args.epochs = epochs
        args.batch_size = batch_size
        args.blr = blr
        # args.class_weights_beta = class_weights_beta
        args.class_weights_beta = 0.99

        if cutoff_epoch == 1:
            args.cutoff_epoch = int(epochs*0.5)
        else:
            args.cutoff_epoch = cutoff_epoch

        model_skill = main(args)
        score = 1 - model_skill
        return score

    print(f"\nRunning Bayesian Optimization for {n_calls} iterations...")
    res_gp = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=10,  # the number of random initialization points
        random_state = 44,
        verbose=True  # Set to False to suppress iteration details
    )

    # --- 5. Print the best hyperparameters found ---
    print("\n--- Bayesian Optimization Results ---")
    print(f"Best validation loss found: {res_gp.fun:.4f}")
    print("Best hyperparameters:")
    print(f" epochs: {res_gp.x[0]}")
    print(f" batch_size: {res_gp.x[1]}")
    print(f" blr: {res_gp.x[2]}")
    print(f" cutoff epoch: {res_gp.x[3]}")

    # # You can also access the best parameters by their names
    # best_epochs, best_batch_size, best_best_blr, best_best_class_weights_beta, best_best_cutoff_epoch= res_gp.x
    # print(f"\nOptimal epochs: {best_epochs}")
    # print(f"Optimal batch_size: {best_batch_size}")
    # print(f"Optimal blr: {best_best_blr}")
    # print(f"Optimal class weights: {best_best_class_weights_beta}")
    # print(f"Optimal cutoff epoch: {best_best_cutoff_epoch}")
    # print("\nBayesian Optimization script finished.")
