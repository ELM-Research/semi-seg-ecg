# Soft boundary-aware Mean Teacher for ECG delineation.
#
# Keeps standard region consistency, and adds a soft boundary consistency term
# from adjacent teacher-probability change, dilated to a +/- band.

# Implemented by: https://github.com/JoeScharpf

from __future__ import annotations

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import utils.lr_sched as lr_sched
import utils.misc as misc
from algorithms import mean_teacher as mt
from utils.boundary_weights import soft_boundary_weight


def train_one_epoch(
    model_student: torch.nn.Module,
    model_teacher: torch.nn.Module,
    labeled_data_loader: Iterable,
    unlabeled_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    use_amp=True,
    config: Optional[dict] = None,
):
    """Mean Teacher + soft boundary consistency."""
    print_freq = 20
    accum_iter = config.get('accum_iter', 1)
    max_norm = config.get('max_norm', None)
    ema_decay = config.get('ema_decay', 0.999)

    boundary_band = int(config.get('boundary_band', 4))
    lambda_region = float(config.get('lambda_region', 1.0))
    lambda_boundary = float(config.get('lambda_boundary', 1.0))
    conf_thresh = float(config.get('conf_thresh', 0.80))
    sample_conf_thresh = float(config.get('sample_conf_thresh', 0.50))

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr',
        misc.SmoothedValue(window_size=1, fmt='{value:.6f}'),
    )
    header = 'Epoch: [{}]'.format(epoch)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    model_student.train()
    model_teacher.eval()
    optimizer.zero_grad()

    num_steps = len(unlabeled_data_loader)
    assert len(labeled_data_loader) == num_steps, \
        "The number of labeled and unlabeled data should be the same"

    for data_iter_step, (labeled, unlabeled) in enumerate(
        metric_logger.log_every(
            zip(
                labeled_data_loader,
                unlabeled_data_loader,
            ),
            print_freq,
            header,
            length=len(labeled_data_loader)
        )
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / num_steps + epoch,
                config,
            )
        ecg_x = labeled['ecg'].to(device, non_blocking=True)
        mask_x = labeled['target'].to(device, non_blocking=True)

        ecg_u_w = unlabeled['ecg'].to(device, non_blocking=True)
        ecg_u_s = unlabeled['ecg_aug'].to(device, non_blocking=True)

        with torch.no_grad():
            pred_u_w = model_teacher(ecg_u_w, return_loss=False)['seg_logits']
            prob_u_w = pred_u_w.softmax(dim=1)
            conf_u_w = prob_u_w.max(dim=1).values  # (B, T)
            w_b = soft_boundary_weight(prob_u_w, band=boundary_band)
            sample_keep = (
                conf_u_w.mean(dim=-1) >= sample_conf_thresh
            ).float().unsqueeze(-1)  # (B, 1)

            # Interiors: strong conf gate, down-weight edges
            w_region = (1.0 - w_b) * (conf_u_w >= conf_thresh).float() * sample_keep
            # Boundaries: soft edge weight; soft conf (do not hard-drop edges)
            w_boundary = w_b * (0.5 + 0.5 * conf_u_w) * sample_keep

        model_student.train()

        num_lb, num_ulb = ecg_x.size(0), ecg_u_w.size(0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model_student(
                torch.cat((ecg_x, ecg_u_s)),
                return_loss=False,
            )
            pred_x, pred_u_s = outputs['seg_logits'].split([num_lb, num_ulb])

            loss_x = F.cross_entropy(pred_x, mask_x)

            # Per-sample CE vs soft teacher targets: (B, T)
            ce = F.cross_entropy(pred_u_s, prob_u_w, reduction='none')

            denom_r = w_region.sum().clamp_min(1.0)
            denom_b = w_boundary.sum().clamp_min(1.0)
            loss_u_region = (ce * w_region).sum() / denom_r
            loss_u_boundary = (ce * w_boundary).sum() / denom_b
            loss_u_s = (
                lambda_region * loss_u_region
                + lambda_boundary * loss_u_boundary
            )

            loss = (loss_x + loss_u_s) / 2.0

        loss_value = loss.item()
        loss_x_value = loss_x.item()
        loss_u_s_value = loss_u_s.item()
        loss_u_region_value = loss_u_region.item()
        loss_u_boundary_value = loss_u_boundary.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model_student.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

            with torch.no_grad():
                for param_q, param_k in zip(
                    model_student.parameters(),
                    model_teacher.parameters(),
                ):
                    param_k.data = (
                        param_k.data * ema_decay
                        + param_q.data * (1.0 - ema_decay)
                    )
                for buffer_q, buffer_k in zip(
                    model_student.buffers(),
                    model_teacher.buffers(),
                ):
                    buffer_k.data = (
                        buffer_k.data * ema_decay
                        + buffer_q.data * (1.0 - ema_decay)
                    )

        torch.cuda.synchronize()

        metric_logger.update(loss_total=loss_value)
        metric_logger.update(loss_x=loss_x_value)
        metric_logger.update(loss_u_s=loss_u_s_value)
        metric_logger.update(loss_u_region=loss_u_region_value)
        metric_logger.update(loss_u_boundary=loss_u_boundary_value)

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_x_value_reduce = misc.all_reduce_mean(loss_x_value)
        loss_u_s_value_reduce = misc.all_reduce_mean(loss_u_s_value)
        loss_u_region_reduce = misc.all_reduce_mean(loss_u_region_value)
        loss_u_boundary_reduce = misc.all_reduce_mean(loss_u_boundary_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (epoch + data_iter_step / num_steps) * 1000
            )
            log_writer.add_scalar('loss_total', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_x', loss_x_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_u_s', loss_u_s_value_reduce, epoch_1000x)
            log_writer.add_scalar(
                'loss_u_region', loss_u_region_reduce, epoch_1000x
            )
            log_writer.add_scalar(
                'loss_u_boundary', loss_u_boundary_reduce, epoch_1000x
            )
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    train_stat = {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }

    return train_stat


def train(config):
    """Reuse Mean Teacher train loop with boundary-aware epoch step."""
    _orig = mt.train_one_epoch
    mt.train_one_epoch = train_one_epoch
    try:
        mt.train(config)
    finally:
        mt.train_one_epoch = _orig


# test() / evaluate entrypoints used by test.py
test = mt.test