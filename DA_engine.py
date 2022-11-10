# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
from random import sample
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import torch.nn.functional as F
import numpy as np
import time


def box_to_mask(boxes, size):
    mask = torch.zeros(size).cuda()
    if boxes == None:
        return mask
    img_w, img_h = size[-1], size[-2]
    img_w, img_h = torch.Tensor([img_w]), torch.Tensor([img_h])
    scale_fct = torch.stack([img_w, img_h, img_w, img_h]).view(1, 4).cuda()
    if len(boxes.size()) != 3:
        boxes = boxes.unsqueeze(0)
    boxes = boxes * scale_fct 
    boxes = boxes[0]
    for box in boxes:
        x, y, w, h = box
        xmin, xmax = x - w/2, x + w/2
        ymin, ymax = y - h/2, y + h/2
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
        mask[:, ymin: ymax, xmin: xmax] = 1
    
    return mask

def swd(source_features, target_features, M=256):
    batch_size = source_features.size(0)
    source_features = source_features.view(-1, source_features.size(-1))
    target_features = target_features.view(-1, target_features.size(-1))
    
    theta = torch.rand(M, 256).cuda() # 256 is the feature dim
    theta = theta / theta.norm(2, dim=1)
    source_proj = theta.mm(source_features.t())
    target_proj = theta.mm(target_features.t())
    source_proj, _ = torch.sort(source_proj, dim=1)
    target_proj, _ = torch.sort(target_proj, dim=1)
    loss = (source_proj - target_proj).pow(2).sum() / M / batch_size
    return loss 

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, args,
        data_loader_src: Iterable, data_loader_tgt: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    prefetcher_src = data_prefetcher(data_loader_src, device, prefetch=True)
    prefetcher_tgt = data_prefetcher(data_loader_tgt, device, prefetch=True)
    samples_src, targets_src = prefetcher_src.next()
    samples_tgt, targets_tgt = prefetcher_tgt.next()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    need_prop_src = torch.ones(1).to(device)
    need_prop_tgt = torch.zeros(1).to(device)
    
    for _ in metric_logger.log_every(range(len(data_loader_src)), print_freq, header):
        outputs_src, domain_outs_src, domain_labels_src, masks_src, hs_src = model(
            samples_src, need_prop_src)
        outputs_tgt, domain_outs_tgt, domain_labels_tgt, masks_tgt, hs_tgt = model(
            samples_tgt, need_prop_tgt)
         
    
        pseudo = None
        out_logits_tgt = outputs_tgt["pred_logits"]
        out_bbox_tgt = outputs_tgt["pred_boxes"]
        
        prob = out_logits_tgt.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits_tgt.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits_tgt.shape[2]
        labels = topk_indexes % out_logits_tgt.shape[2]
        
        boxes = torch.gather(out_bbox_tgt, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        indices_object = (labels != 0) # remote background predictions
        labels = labels[indices_object]
        scores = scores[indices_object]
        boxes = boxes[indices_object]
        
        scores_indices = (scores > 0.5) # Pseudo label selection using confidence
        
        if scores_indices.sum():
            pseudo = {'boxes': boxes[scores_indices],
                    'labels': labels[scores_indices],
                    'image_id': targets_tgt[0]['image_id'],
                    'orig_size': targets_tgt[0]['orig_size'],
                    'size': targets_tgt[0]['size']}        
       
        loss_dict, _ = criterion(outputs_src, targets_src)
        weight_dict = criterion.weight_dict
        
        loss_da = 0
        loss_global_da = 0
        for l in range(len(domain_outs_src)):
            domain_out_src = domain_outs_src[l]
            domain_label_src = domain_labels_src[l]
            mask_src = masks_src[l]
            domain_out_tgt = domain_outs_tgt[l]
            domain_label_tgt = domain_labels_tgt[l]
            mask_tgt = masks_tgt[l]
            
            domain_prob_src = F.log_softmax(domain_out_src, dim=1)
            domain_prob_tgt = F.log_softmax(domain_out_tgt, dim=1)
            DA_img_loss_src = F.nll_loss(
                domain_prob_src, domain_label_src, reduction="none")
            DA_img_loss_tgt = F.nll_loss(
                domain_prob_tgt, domain_label_tgt, reduction="none")
            mask_src = ~mask_src
            mask_tgt = ~mask_tgt
            
            global_DA_img_loss = DA_img_loss_src.mul(mask_src).sum()/mask_src.sum() + DA_img_loss_tgt.mul(mask_tgt).sum()/mask_tgt.sum()

            # Mask out background regions
            mask_ins_src = box_to_mask(targets_src[0]['boxes'], mask_src.size())
            mask_ins_src += 1 # for numeric stability 
            mask_ins_src = mask_ins_src / mask_ins_src.mean()
            mask_final_src = mask_src * mask_ins_src
            if pseudo is None:
                mask_ins_tgt = box_to_mask(None, mask_tgt.size())
            else:
                mask_ins_tgt = box_to_mask(pseudo['boxes'], mask_tgt.size())
            mask_ins_tgt += 1 # for numeric stability
            mask_ins_tgt = mask_ins_tgt / mask_ins_tgt.mean()
            mask_final_tgt = mask_tgt * mask_ins_tgt
    
            if mask_final_tgt.sum() and mask_final_src.sum():
                DA_img_loss = DA_img_loss_src.mul(mask_final_src).sum()/mask_final_src.sum() + DA_img_loss_tgt.mul(mask_final_tgt).sum()/mask_final_tgt.sum()
            elif mask_final_src.sum():
                DA_img_loss = DA_img_loss_src.mul(mask_final_src).sum()/mask_final_src.sum() + DA_img_loss_tgt.mul(mask_tgt).sum()/mask_tgt.sum()
            elif mask_final_tgt.sum():
                DA_img_loss = DA_img_loss_src.mul(mask_src).sum()/mask_src.sum() + DA_img_loss_tgt.mul(mask_final_tgt).sum()/mask_final_tgt.sum()
            else:
                DA_img_loss = DA_img_loss_src.mul(mask_src).sum()/mask_src.sum() + DA_img_loss_tgt.mul(mask_tgt).sum()/mask_tgt.sum()
            
            loss_da += DA_img_loss
            loss_global_da += global_DA_img_loss
        loss_dict["loss_da"] = args.instance_loss_coef * loss_da + loss_global_da
        loss_dict["loss_wasserstein"] = swd(hs_src[-1], hs_tgt[-1])
        
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)
       
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(
                model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples_src, targets_src = prefetcher_src.next()
        samples_tgt, targets_tgt = prefetcher_tgt.next()
        if samples_tgt is None:
            prefetcher_tgt = data_prefetcher(data_loader_tgt, device, prefetch=True)
            samples_tgt, targets_tgt = prefetcher_tgt.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    it = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict, _ = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
      
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
  
        results = postprocessors['bbox'](outputs, orig_target_sizes) 
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](
                results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target,
               output in zip(targets, results)} 
     
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator

