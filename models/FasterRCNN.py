from config import BackboneConfig, RPNConfig, ROIBoxHeadConfig # TODO FIX CONFIGS
from models.common import get_backbone, get_rpn
from torch import nn

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from typing import Dict, Optional, Tuple


def get_fastercnn_roi_head(
        output_pool_size=7,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None
):
    """
    Builds RoI head classification and regression network for FasterRCNN.

    Args:
        output_pool_size (int): spatial resolution of the feature map pooled out of each region proposal
        score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        nms_thresh (float): NMS threshold for the prediction head. Used during inference
        detections_per_img (int): maximum number of detections per image, for all classes.
        fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Returns:
        roi_heads (nn.Module): RoI classification/regression head
    """
    # RoI Pooling layer
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=output_pool_size,
                                      sampling_ratio=2)
    # Pool Layer -> flat features
    box_head = TwoMLPHead(256 * output_pool_size ** 2, 1024)
    # flat features -> classes, regression
    box_predictor = FastRCNNPredictor(1024, 2)
    # wrap in a module
    roi_heads = RoIHeads(
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_thresh=fg_iou_thresh,
        bg_iou_thresh=bg_iou_thresh,
        batch_size_per_image=batch_size_per_image,
        positive_fraction=positive_fraction,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        bbox_reg_weights=bbox_reg_weights,
    )
    return roi_heads


class FasterRCNN(GeneralizedRCNN, nn.Module):
    def __init__(
            self,
            backbone_config: Optional[Dict] = None,
            rpn_config: Optional[Dict] = None,
            roi_box_head_config: Optional[Dict] = None
    ):
        # set up configs
        if backbone_config is None:
            backbone_config = BackboneConfig()._to_dict()
        if rpn_config is None:
            rpn_config = RPNConfig()._to_dict()
        if roi_box_head_config is None:
            roi_box_head_config = ROIBoxHeadConfig()._to_dict()

        # construct components from configuration
        backbone = get_backbone(**backbone_config)
        rpn = get_rpn(**rpn_config)
        roi_head = get_fastercnn_roi_head(**roi_box_head_config)
        transform = GeneralizedRCNNTransform(128, 128, [0], [1])
        # build the final model
        super().__init__(backbone, rpn, roi_head, transform)