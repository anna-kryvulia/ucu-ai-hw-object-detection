"""RT-DETR (Hugging Face transformers) behind the torchvision detector interface.

Same code as the model cell of notebook 05, moved here so the own pipeline (notebook 07) can rebuild
the trained model and load its checkpoint:

    model = load_weights(build_rtdetr(load_config("rtdetr_baseline")), Run("rtdetr_baseline", None, P), "best", dev)
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")          # download bars are saved as widgets
os.environ.setdefault("HF_HUB_VERBOSITY", "error")                  # no notice about downloading without an HF token

import torch
import transformers
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_convert
from transformers import RTDetrForObjectDetection

from .data.classes import CLASS_NAMES, NUM_CLASSES

transformers.utils.logging.disable_progress_bar()
HF_KWARGS = ["num_queries", "num_denoising", "matcher_class_cost", "matcher_bbox_cost", "matcher_giou_cost",
             "auxiliary_loss"]
PRIOR_BIAS = -math.log((1 - 0.01) / 0.01)


def group_losses(loss_dict):
    """RT-DETR returns ~40 weighted terms (last layer, each other decoder layer, the encoder, denoising);
    they are summed per kind, so the total is unchanged: loss_vfl / loss_bbox / loss_giou + dn_ for denoising."""
    out = {}
    for k, v in loss_dict.items():
        kind = ("dn_" if "_dn_" in k else "") + k.split("_aux_")[0].split("_dn_")[0]
        out[kind] = out.get(kind, 0) + v
    return out


class RTDETR(nn.Module):
    """Images in pixels -> RT-DETR -> losses (training) or boxes in pixels (inference), like torchvision."""

    def __init__(self, hf, min_size, max_size, backbone_lr_mult):
        super().__init__()
        self.hf = hf
        self.transform = GeneralizedRCNNTransform(min_size, max_size, [0.0] * 3, [1.0] * 3)   # pixels in 0-1
        self.backbone_lr_mult = backbone_lr_mult

    def param_groups(self, lr):
        backbone = [p for p in self.hf.model.backbone.parameters() if p.requires_grad]
        ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return [{"params": backbone, "lr": lr * self.backbone_lr_mult}, {"params": rest, "lr": lr}]

    def forward(self, images, targets=None):
        original = [img.shape[-2:] for img in images]
        il, targets = self.transform(images, targets)
        x = il.tensors
        scale = torch.tensor([x.shape[-1], x.shape[-2]] * 2, device=x.device, dtype=torch.float32)
        if self.training:                                             # boxes: centre + size, 0-1 of the padded input
            labels = [{"class_labels": t["labels"] - 1, "boxes": box_convert(t["boxes"], "xyxy", "cxcywh") / scale}
                      for t in targets]
            return group_losses(self.hf(pixel_values=x, labels=labels).loss_dict)
        out = self.hf(pixel_values=x)
        prob = out.logits.sigmoid()                                   # [images, queries, classes]
        n_cls = prob.shape[-1]
        scores, idx = prob.flatten(1).topk(min(self.hf.config.num_queries, prob[0].numel()))
        query, cls = idx // n_cls, idx % n_cls                        # best (query, class) pairs, no NMS
        boxes = box_convert(out.pred_boxes, "cxcywh", "xyxy") * scale
        boxes = boxes.gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
        dets = []
        for b, s, c, (h, w) in zip(boxes, scores, cls, il.image_sizes):
            b = torch.stack([b[:, 0].clamp(0, w), b[:, 1].clamp(0, h), b[:, 2].clamp(0, w), b[:, 3].clamp(0, h)], 1)
            dets.append({"boxes": b, "scores": s, "labels": c + 1})   # labels 1-8 as for the CNN detectors
        return self.transform.postprocess(dets, il.image_sizes, original)


def new_head(d):
    head = nn.Linear(d, NUM_CLASSES)
    nn.init.xavier_uniform_(head.weight)
    nn.init.constant_(head.bias, PRIOR_BIAS)
    return head


def build_rtdetr(cfg):
    """RT-DETR with COCO weights and new 8-class heads (the trained weights are loaded afterwards)."""
    m = cfg["model"]
    hf = RTDetrForObjectDetection.from_pretrained(m["arch"], anchor_image_size=None, eval_size=None,
                                                  **{k: m[k] for k in HF_KWARGS})   # anchors from the real input size
    d = hf.config.d_model
    hf.config.num_labels = NUM_CLASSES
    hf.config.id2label = dict(enumerate(CLASS_NAMES))
    hf.config.label2id = {n: i for i, n in enumerate(CLASS_NAMES)}
    hf.model.enc_score_head = new_head(d)
    hf.model.decoder.class_embed = nn.ModuleList([new_head(d) for _ in range(hf.config.decoder_layers)])
    if hasattr(hf, "class_embed"):                                    # older transformers keep a second reference
        hf.class_embed = hf.model.decoder.class_embed
    hf.model.denoising_class_embed = nn.Embedding(NUM_CLASSES + 1, d, padding_idx=NUM_CLASSES)
    nn.init.xavier_uniform_(hf.model.denoising_class_embed.weight)
    return RTDETR(hf, m["min_size"], m["max_size"], m["backbone_lr_mult"])
