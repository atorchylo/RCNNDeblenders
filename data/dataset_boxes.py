from .base.dataset import DatasetBase
from .base.types import TargetDict, ImageTargetTuple
from .base.normalize import Normalizer, LinearNormCosmos

from typing import Any
import numpy as np
import torch
import os


class DatasetBoxes(DatasetBase):
    def __init__(self, file_path: str, normalizer: Normalizer = LinearNormCosmos()):
        self._prepare_filenames(file_path)
        self.normalizer = normalizer

    def _prepare_filenames(self, file_path):
        """Parses through file names in the input directory"""
        file_names = os.listdir(file_path)
        image_files = [file for file in file_names if file.startswith('image')]
        target_files = [file for file in file_names if file.startswith('target')]
        assert len(image_files) == len(target_files), 'there must be one target for each image'
        # sort files
        sort_func = lambda name: int(name.split('_')[-1].replace('.npy', ''))
        image_files = sorted(image_files, key=sort_func)
        target_files = sorted(target_files, key=sort_func)
        # append directory
        self.image_files = [os.path.join(file_path, file) for file in image_files]
        self.target_files = [os.path.join(file_path, file) for file in target_files]

    def load_image(self, index: int) -> Any:
        img = np.load(self.image_files[index])
        return img

    def load_target_dict(self, index: int) -> TargetDict:
        boxes = np.load(self.target_files[index])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
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
        return len(self.image_files)
