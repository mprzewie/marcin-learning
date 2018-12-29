from typing import Sequence, List, Callable, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from pathlib import Path

from torchvision.transforms import transforms

from PIL import Image


class COCOStuff(Dataset):
    def __init__(
            self,
            images_path: Path,
            annotations_json: Path,
            transformations: List[Callable] = None,
            target_transformations: List[Callable] = None
    ):
        if transformations is None:
            transformations = []
        if target_transformations is None:
            target_transformations = []
        self.images_path = images_path
        self.coco = COCO(annotations_json)
        self.transformations = transforms.Compose(transformations)
        self.target_transformations = transforms.Compose(target_transformations)

    def __getitem__(self, index: int):
        image, mask = self.get_image_and_mask(self._image_id(index))
        return self.transformations(image), self.target_transformations(mask)

    def __len__(self):
        return len(self.image_ids)

    @property
    def n_classes(self) -> int:
        return max([a["category_id"] for a in self.coco.anns.values()]) + 1

    @property
    def image_ids(self) -> List[int]:
        return self.coco.getImgIds()

    def get_image_and_mask(self, image_id: int) -> Tuple[Image.Image, np.ndarray]:
        return self._get_pil_image(image_id), self._get_np_segmask(image_id)

    def _get_pil_image(self, image_id: int) -> Image.Image:
        return Image.open(self._image_path(image_id))

    def _get_np_segmask(self, image_id: int) -> np.ndarray:
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        binary_masks = np.array([mask_utils.decode(a["segmentation"]) * a["category_id"] for a in annotations])
        return binary_masks.max(axis=0)

    def _image_path(self, image_id: int) -> Path:
        return self.images_path / self._file_name(image_id)

    def _image_id(self, index: int):
        return self.image_ids[index]

    def _file_name(self, image_id: int) -> str:
        return self.coco.imgs[image_id]["file_name"]
