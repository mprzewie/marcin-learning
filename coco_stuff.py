from abc import abstractmethod
from typing import Sequence, List, Callable, Tuple, Dict

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from pathlib import Path

from torchvision.transforms import transforms

from PIL import Image

_COCO_CATEGORY_ID_KEY = "category_id"
_COCO_FILE_NAME_KEY = "file_name"
_COCO_CATEGORY_NAME_KEY = "name"
_COCO_SUPERCATEGORY_NAME_KEY = "supercategory"


class _COCOStuff(Dataset):
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
    def category_to_class(self) -> Dict[int, int]:
        cat_name_to_class = {cat: i for (i, cat) in enumerate(sorted(set(self.categories_names.values())))}
        return {cat: cat_name_to_class[self.categories_names[cat]] for cat in self.categories_names.keys()}

    @property
    def class_to_category(self) -> Dict[int, int]:
        return {v: k for (k, v) in self.category_to_class.items()}

    @property
    def _category_class_array(self) -> np.ndarray:
        max_category = max(self.category_to_class.keys())
        return np.array([self.category_to_class.get(cat, 0) for cat in range(max_category + 1)])

    @property
    def n_classes(self) -> int:
        return len(np.unique(self._category_class_array))

    @property
    def image_ids(self) -> List[int]:
        return self.coco.getImgIds()

    def get_image_and_mask(self, image_id: int) -> Tuple[Image.Image, np.ndarray]:
        return self._get_pil_image(image_id), self._get_segmask(image_id)

    def _get_pil_image(self, image_id: int) -> Image.Image:
        return Image.open(self._image_path(image_id)).convert("RGB")

    def _get_segmask(self, image_id: int) -> np.ndarray:
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        if len(annotations) > 0:
            binary_masks = np.array([
                self.coco.annToMask(a) * a[_COCO_CATEGORY_ID_KEY] for a in annotations
            ])
        else:
            binary_masks = np.zeros([1, *self._get_pil_image(image_id).size])

        mask = binary_masks.max(axis=0)
        mask[mask == 0] = self._background_category
        return self._category_class_array[mask]

    def _image_path(self, image_id: int) -> Path:
        return self.images_path / self._file_name(image_id)

    def _image_id(self, index: int):
        return self.image_ids[index]

    def _file_name(self, image_id: int) -> str:
        return self.coco.imgs[image_id][_COCO_FILE_NAME_KEY]

    @property
    def _background_category(self):
        return max(self.categories_names.keys())

    @property
    def _categories_and_supercategories(self) -> Dict[int, Dict[str, str]]:
        return {
            cat["id"]: {key: cat[key] for key in [_COCO_CATEGORY_NAME_KEY, _COCO_SUPERCATEGORY_NAME_KEY]}
            for cat in self.coco.loadCats(self.coco.getCatIds())
        }

    @property
    def categories_names(self) -> Dict[int, str]:
        """Each unique name is a separate class"""
        return self._category_to_readable(_COCO_CATEGORY_NAME_KEY)

    def _category_to_readable(self, name_key: str):
        return {cat: cas[name_key] for cat, cas in self._categories_and_supercategories.items()}

    @property
    def classes_names(self) -> Dict[int, str]:
        return {cls: self.categories_names[self.class_to_category[cls]] for cls in range(self.n_classes)}


class COCOStuffSuper(_COCOStuff):
    @property
    def categories_names(self) -> Dict[int, str]:
        """Category ids are mapped to supercategory names, which will result in a lower number of ML classes
        """
        return self._category_to_readable(_COCO_SUPERCATEGORY_NAME_KEY)
