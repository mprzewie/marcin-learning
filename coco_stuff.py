from typing import Sequence, List, Callable, Tuple, Dict

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
            target_transformations: List[Callable] = None,
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
    def _categories_ids(self) -> Dict[int, int]:
        return {
            cat_id: i
            for i, cat_id in enumerate(self.coco.getCatIds())
        }

    @property
    def _background_id(self):
        return max(self.coco.getCatIds())

    @property
    def n_classes(self) -> int:
        return len(self._categories_ids)

    @property
    def image_ids(self) -> List[int]:
        return self.coco.getImgIds()

    def get_image_and_mask(self, image_id: int) -> Tuple[Image.Image, np.ndarray]:
        return self._get_pil_image(image_id), self._get_np_segmask(image_id)

    def _get_pil_image(self, image_id: int) -> Image.Image:
        return Image.open(self._image_path(image_id)).convert("RGB")

    def _get_np_segmask(self, image_id: int) -> np.ndarray:
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        if len(annotations) > 0:
            binary_masks = np.array([
                self.coco.annToMask(a) * a["category_id"] for a in annotations
            ])
        else:
            binary_masks = np.zeros([1, *self._get_pil_image(image_id).size])

        mask = binary_masks.max(axis=0)
        mask[mask == 0] = self._background_id
        return mask

    def _image_path(self, image_id: int) -> Path:
        return self.images_path / self._file_name(image_id)

    def _image_id(self, index: int):
        return self.image_ids[index]

    def _file_name(self, image_id: int) -> str:
        return self.coco.imgs[image_id]["file_name"]
