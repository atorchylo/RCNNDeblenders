import torch
from src.models.common import get_backbone, get_rpn, AVAILABLE_BACKBONES
from src.models.FasterRCNN import get_fastercnn_roi_head
from torchvision.models.detection.image_list import ImageList

from collections import OrderedDict

# fix seed for testing
torch.manual_seed(0)


### TEST COMPONENTS ###
def test_backbones():
    for name in AVAILABLE_BACKBONES:
        backbone = get_backbone(name)
        # test first layer
        assert str(
            backbone.body.conv1) == 'Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)'
        # test shapes
        input = torch.randn((1, 3, 128, 128))
        out = backbone(input)
        correct_shapes = {
            '0': [1, 256, 32, 32],
            '1': [1, 256, 16, 16],
            '2': [1, 256, 8, 8],
            '3': [1, 256, 4, 4],
            'pool': [1, 256, 2, 2]
        }
        assert out.keys() == correct_shapes.keys()
        for key in correct_shapes:
            assert list(out[key].shape) == correct_shapes[key]


def test_rpn():
    # build RPN
    backbone = get_backbone('resnet50', 6)
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn = get_rpn(anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios, post_nms_top_n_train=20)

    # build input
    img = torch.randn(1, 6, 128, 128)
    img_list = ImageList(img, [(128, 128)])
    features = backbone(img)
    targets = dict(
        boxes=torch.tensor([[1, 1, 5, 5], [120, 120, 350, 350]], dtype=torch.float32),
        labels=torch.tensor([1, 1], dtype=torch.int64)
    )

    boxes, losses = rpn(img_list, features, [targets])

    assert list(boxes[0].shape) == [20, 4]
    assert 'loss_objectness' in losses
    assert 'loss_rpn_box_reg' in losses


def test_fastercnn_roi_head():
    roi_heads = get_fastercnn_roi_head()

    features = OrderedDict()
    features['0'] = torch.rand(1, 256, 32, 32)
    features['1'] = torch.rand(1, 256, 16, 16)
    features['2'] = torch.rand(1, 256, 8, 8)
    features['3'] = torch.rand(1, 256, 4, 4)
    features['pool'] = torch.rand(1, 256, 2, 2)

    # create boxes
    boxes = torch.rand(100, 4) * 128
    boxes[:, 2:] += boxes[:, :2]
    target_boxes = torch.rand(6, 4) * 128
    target_boxes[:, 2:] += target_boxes[:, :2]
    # original image size, before computing the feature maps
    image_sizes = [(128, 128)]
    targets = dict(
        boxes=target_boxes,
        labels=torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int64)
    )
    # test training forward
    roi_heads.train()
    results, losses = roi_heads(features, [boxes], image_sizes, [targets])
    assert results == []
    assert 'loss_classifier' in losses
    assert 'loss_box_reg' in losses
    # test inference
    roi_heads.eval()
    results, losses = roi_heads(features, [boxes], image_sizes, [targets])
    assert 'boxes' in results[0]
    assert 'labels' in results[0]
    assert 'scores' in results[0]


if __name__ == "__main__":
    test_backbones()
    test_rpn()
    test_fastercnn_roi_head()

