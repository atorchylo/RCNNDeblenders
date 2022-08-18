from ..utils.ellipse_utils import moment
from .base.dataset import DatasetBase
from .base.types import TargetDict, ImageTargetTuple
from .base.normalize import Normalizer, LinearNormCosmos
from btk.utils import load_blend_results
from typing import Any
import torch


class CosmosHSC(DatasetBase):
    def __init__(self, path: str, normalizer: Normalizer = LinearNormCosmos):
        data = load_blend_results(path, 'HSC')
        self.images = data['blend_images']
        self.isolated = data['isolated_images']
        self.blend_list = data['blend_list']
        self.normalizer = normalizer()
        self.i_band_idx = 2  # needed for moments calculation

    def load_image(self, index: int) -> Any:
        return self.images[index]

    def load_target_dict(self, index: int) -> TargetDict:
        num_objs = len(self.blend_list[index])

        boxes = []
        for i in range(num_objs):
            # compute moments
            galaxy = self.isolated[index, i, self.i_band_idx]
            Ixx = moment(galaxy, 2, 0)
            Iyy = moment(galaxy, 0, 2)
            x, y = self.blend_list[index][i]['x_peak', 'y_peak']
            x1, x2 = x - Ixx, x + Ixx
            y1, y2 = y - Iyy, y + Iyy
            boxes.append((x1, y1, x2, y2))
        boxes = torch.tensor(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = TargetDict(
            boxes=boxes,
            labels=labels,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd
        )
        return target

    def _transform_all(self, image: Any, target_dict: TargetDict) -> ImageTargetTuple:
        image_normalized = self.normalizer.forward(image)
        image_tensor = torch.tensor(image_normalized, dtype=torch.float32)
        return image_tensor, target_dict

    def __len__(self) -> int:
        return self.images.shape[0]
