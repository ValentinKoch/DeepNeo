from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class InferenceDataset(Dataset):
    def __init__(self, images: List[Path], num_classes=4) -> None:
        self.images = images
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:

        image_path = Path(self.images[idx])
        result = {}
        im = (
            Image.open(image_path)
            .resize(
                (config.IMAGE_SIZE_SEGMENTATION, config.IMAGE_SIZE_SEGMENTATION),
                Image.Resampling.NEAREST,
            )
            .convert("L")
        )
        result["image"] = torch.unsqueeze(torch.Tensor(np.array(im) / 255.0), 0)
        result["filename"] = str(image_path.stem)
        result["index"] = idx

        return result
