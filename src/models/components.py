from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign


AVAILABLE_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
BACKBONE_OUT_CHANNELS = 256


def get_backbone(name='resnet50', input_channels=None):
    """
    Builds backbone network.

    Args:
        name (str): name of the backbone, must be one of ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        input_channels (int): number of channels in the input image.

    Returns:
        backbone (nn.Module): returns the backbone of a neural network.
    """
    assert name in AVAILABLE_BACKBONES, f'name must be one of {AVAILABLE_BACKBONES}'
    backbone = resnet_fpn_backbone(backbone_name=name, weights='DEFAULT', trainable_layers=5)
    if input_channels is not None:
        backbone.body.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return backbone


def get_rpn(
        # Anchor parameters
        anchor_sizes=((4,), (8,), (16,), (32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        # RPN parameters
        pre_nms_top_n_train=2000,
        pre_nms_top_n_test=1000,
        post_nms_top_n_train=2000,
        post_nms_top_n_test=1000,
        nms_thresh=0.7,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=128,
        positive_fraction=0.5,
        score_thresh=0.0
):
    """
    Builds RegionProposalNetwork from configuration.

    anchor_sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps in the backbone.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        anchor_sizes (Tuple[Tuple[int]]): base sizes of anchor boxes to use in region proposal
        aspect_ratios (Tuple[Tuple[float]]): ratios w/h for each base size of the anchor box
        backbone_out_channles (int): number of channels for each feature map outputted by the backbone
        pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh

    Returns:
        rpn (nn.Module): Region Proposal Network
    """
    assert len(anchor_sizes) == 5, "number of scales must correspond to the number of feature maps"
    assert len(aspect_ratios) == 5, "number of scales must correspond to the number of feature maps"
    # maps anchors on the input images
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    # anchor feature vector to class regression
    rpn_head = RPNHead(BACKBONE_OUT_CHANNELS, rpn_anchor_generator.num_anchors_per_location()[0])
    # wrap in a module
    rpn = RegionProposalNetwork(
        anchor_generator=rpn_anchor_generator, head=rpn_head,
        fg_iou_thresh=fg_iou_thresh, bg_iou_thresh=bg_iou_thresh,
        batch_size_per_image=batch_size_per_image,
        positive_fraction=positive_fraction,
        pre_nms_top_n=dict(training=pre_nms_top_n_train, testing=pre_nms_top_n_test),  # TODO tune number of boxes
        post_nms_top_n=dict(training=post_nms_top_n_train, testing=post_nms_top_n_test),  # TODO tune number of boxes
        nms_thresh=nms_thresh,
        score_thresh=score_thresh,
    )
    return rpn


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
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=output_pool_size, sampling_ratio=2)
    # Pool Layer -> flat features
    box_head = TwoMLPHead(BACKBONE_OUT_CHANNELS * output_pool_size ** 2, 1024)
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