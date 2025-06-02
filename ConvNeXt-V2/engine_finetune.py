# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate
from torchmetrics.segmentation import MeanIoU
import sys

sys.path.append("..")
from unbalanced_loss import get_class_balanced_weights


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    metric = MeanIoU(num_classes=4, per_class=True, input_format='index')
    for data_iter_step, (samples, targets, domain_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = targets.unsqueeze(1) # add a channel dimension [B, 1, 512, 512]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples).permute(0, 2, 3, 1).contiguous()
                output_ = output.view(-1, output.size(3))

                targets = targets.permute(0, 2, 3, 1).contiguous()
                targets_ = targets.view(-1, targets.size(3)).squeeze(1).long()
                loss = criterion(output_, targets_)
        else: # full precision
            # samples of shape [B, 3, 512, 512]
            output = model(samples)
            # output of shape [B, 4, 512, 512] # 4 is the number of classes
            output = output.permute(0, 2, 3, 1).contiguous()
            # output of shape [B, 512, 512, 4]  channels in last dimension
            output_ = output.view(-1, output.size(3))
            # output_ of shape [B * 512 * 512, 4] # everything else is flattened and channels in last dimension

            targets_ = targets.permute(0, 2, 3, 1).contiguous()
            # targets_ of shape [B, 512, 512, 1]   
            targets_ = targets_.view(-1, targets_.size(3)).squeeze(1).long()
            # targets_ of shape [B * 512 * 512]
            loss = criterion(output_, targets_)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        output = output.permute(0, 3, 1, 2)
        output = torch.nn.functional.softmax(output, dim=1)
        output = output.argmax(dim=1)
        targets = targets.squeeze(1)
        score = metric(output.cpu(), targets.cpu())


        # score is per class.

        metric_logger.update(loss=loss_value)
        metric_logger.meters['mIoU_class_0'].update(score[0].item())
        metric_logger.meters['mIoU_class_1'].update(score[1].item())
        metric_logger.meters['mIoU_class_2'].update(score[2].item())
        metric_logger.meters['mIoU_class_3'].update(score[3].item())


        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(mIoU_class_0=score[0].item(), head="loss")
            log_writer.update(mIoU_class_1=score[1].item(), head="loss")
            log_writer.update(mIoU_class_2=score[2].item(), head="loss")
            log_writer.update(mIoU_class_3=score[3].item(), head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    metric_values = metric.compute()
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict['mIoU_class_0'] = metric_values[0].item()
    return_dict['mIoU_class_1'] = metric_values[1].item()
    return_dict['mIoU_class_2'] = metric_values[2].item()
    return_dict['mIoU_class_3'] = metric_values[3].item()
    return return_dict

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    metric = MeanIoU(num_classes=4, per_class=True, input_format='index')
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        domain_label = batch[2]


        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target.unsqueeze(1) # add a channel dimension [B, 1, 512, 512]

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            output = model(images).permute(0, 2, 3, 1).contiguous()
            output_ = output.view(-1, output.size(3))

            target = target.permute(0, 2, 3, 1).contiguous()
            target_ = target.view(-1, target.size(3)).squeeze(1).long()
            loss = criterion(output_, target_)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        
        output = output.permute(0, 3, 1, 2)
        output = torch.nn.functional.softmax(output, dim=1)
        output = output.argmax(dim=1)
        target = target.squeeze()


        score = metric(output.cpu(), target.cpu())
        metric_logger.meters['mIoU_class_0'].update(score[0].item())
        metric_logger.meters['mIoU_class_1'].update(score[1].item())
        metric_logger.meters['mIoU_class_2'].update(score[2].item())
        metric_logger.meters['mIoU_class_3'].update(score[3].item())


    test_metric = metric.compute()
    # print(test_metric)
    logging_text = "**** "
    logging_text += f"mIoU Class 0: {test_metric[0].item():.3f}\n"
    logging_text += f"mIoU Class 1: {test_metric[1].item():.3f}\n"
    logging_text += f"mIoU Class 2: {test_metric[2].item():.3f}\n"
    logging_text += f"mIoU Class 3: {test_metric[3].item():.3f}\n"

    logging_text += f"loss {metric_logger.loss.global_avg:.3f}\n"

    print(logging_text)
    
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # we replace the metrics with metric.compute() values
    return_dict['mIoU_class_0'] = test_metric[0].item()
    return_dict['mIoU_class_1'] = test_metric[1].item()
    return_dict['mIoU_class_2'] = test_metric[2].item()
    return_dict['mIoU_class_3'] = test_metric[3].item()
    return return_dict