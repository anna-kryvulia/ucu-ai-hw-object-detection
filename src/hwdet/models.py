"""Rebuild trained torchvision detectors from their configs, to load saved checkpoints (own pipeline).

Same architecture code as notebook 02 (Faster R-CNN). Pretrained weights are not downloaded here:
every weight comes from the checkpoint loaded afterwards with engine.load_weights.

    model = build_frcnn(load_config("frcnn_baseline"))
    model = load_weights(model, Run("frcnn_baseline", None, P), "best", device)
"""
from __future__ import annotations

from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from .data.classes import NUM_CLASSES

FRCNN_ARCHS = {"resnet50_fpn": (fasterrcnn_resnet50_fpn, 1), "resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, 2)}
FRCNN_KWARGS = ["min_size", "max_size", "rpn_pre_nms_top_n_train", "rpn_post_nms_top_n_train",
                "rpn_pre_nms_top_n_test", "rpn_post_nms_top_n_test", "box_batch_size_per_image",
                "box_positive_fraction", "box_detections_per_img", "box_score_thresh"]


def build_frcnn(cfg: dict):
    """Faster R-CNN with the anchors, RPN / RoI settings and 8-class head of a config (no weights)."""
    m = cfg["model"]
    ctor, rpn_conv_depth = FRCNN_ARCHS[m["arch"]]
    model = ctor(weights=None, weights_backbone=None, **{k: m[k] for k in FRCNN_KWARGS if k in m})
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    sizes = tuple((s,) for s in m["anchor_sizes"])
    model.rpn.anchor_generator = AnchorGenerator(sizes, (tuple(m["aspect_ratios"]),) * len(sizes))
    if len(m["aspect_ratios"]) != model.rpn.head.cls_logits.out_channels:
        model.rpn.head = RPNHead(256, len(m["aspect_ratios"]), conv_depth=rpn_conv_depth)
    return model
