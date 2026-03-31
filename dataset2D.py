import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple

class MedicalImageDataset2D(Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 phase: str = 'train',
                 image_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 use_hist_eq: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.image_size = image_size
        self.normalize = normalize
        self.use_hist_eq = use_hist_eq

        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")

        self.image_files = []
        for ext in self.supported_extensions:
            self.image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")

        self.valid_files = []
        
        # Filter corrupt files
        for img_file in self.image_files:
            image_path = os.path.join(self.image_dir, img_file)
            if not self._is_valid_image(image_path):
                print(f"[Warning] Skip corrupted image: {img_file}")
                continue

            try:
                mask_path = self._find_mask_path(img_file)
            except ValueError:
                continue

            if not self._is_valid_image(mask_path):
                print(f"[Warning] Skip corrupted mask: {os.path.basename(mask_path)}")
                continue

            self.valid_files.append(img_file)

        if len(self.valid_files) == 0:
            raise ValueError("No valid image-mask pairs after filtering!")

        self.transforms = self._get_transforms()

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False

    def _get_transforms(self):
        if self.phase == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2, 0.2),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('L')
            image_np = np.asarray(image)
            if self.use_hist_eq:
                image_np = cv2.equalizeHist(image_np)
            return image_np
        except Exception as e:
            raise ValueError(f"Cannot load image: {image_path}, Error: {e}")

    def _load_mask(self, mask_path: str) -> np.ndarray:
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.asarray(mask)
            mask = (mask > 127).astype(np.uint8)
            return mask
        except Exception as e:
            raise ValueError(f"Cannot load mask: {mask_path}, Error: {e}")

    def _find_mask_path(self, img_file: str) -> str:
        base_name, ext = os.path.splitext(img_file)
        candidates = [
            img_file,
            *[f"{base_name}_mask{e}" for e in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')],
            *[f"{base_name}{e}" for e in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')],
            *[f"{base_name}-1{e}" for e in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')],
            *[f"{base_name}_segmentation{e}" for e in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
        ]
        for mask_file in candidates:
            mask_path = os.path.join(self.mask_dir, mask_file)
            if os.path.exists(mask_path):
                return mask_path

        all_masks = [f for f in os.listdir(self.mask_dir) if f.lower().endswith(tuple(self.supported_extensions))]
        for mf in all_masks:
            if base_name in mf:
                return os.path.join(self.mask_dir, mf)

        raise ValueError(f"Cannot find mask for {img_file}")

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int):
        filename = self.valid_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        mask_path = self._find_mask_path(filename)

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        if self.phase == 'train':
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed); np.random.seed(seed)
            image_tensor = self.transforms(image_pil)

            torch.manual_seed(seed); np.random.seed(seed)
            mask_tensor = transforms.Compose([
                transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10, transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])(mask_pil)
        else:
            image_tensor = self.transforms(image_pil)
            mask_tensor = transforms.Compose([
                transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])(mask_pil)

        mask_tensor = (mask_tensor > 0.5).long().squeeze(0)

        if self.normalize:
            image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-8)

        return image_tensor, mask_tensor, filename
