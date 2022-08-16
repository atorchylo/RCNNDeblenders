import torch
from src.models.components import get_backbone, get_rpn, AVAILABLE_BACKBONES
from torchvision.models.detection.image_list import ImageList

from collections import OrderedDict


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
    rpn = get_rpn(anchor_sizes, aspect_ratios, post_nms_top_n_train=20)

    # build input
    img = torch.randn(1, 6, 128, 128)
    img_list = ImageList(img, [(128, 128)])
    features = backbone(img)
    targets = OrderedDict(
        boxes=torch.tensor([[1, 1, 5, 5], [120, 120, 350, 350]], dtype=torch.float32),
        labels=torch.tensor([1, 1], dtype=torch.int64)
    )

    boxes, losses = rpn(img_list, features, [targets])
    assert list(boxes[0].shape) == [20, 4]
    assert 'loss_objectness' in losses
    assert 'loss_rpn_box_reg' in losses


if __name__ == "__main__":
    test_backbones()
    test_rpn()
