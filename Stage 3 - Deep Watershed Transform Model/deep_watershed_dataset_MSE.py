import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from pathlib import Path
import glob

class DeepWatershedDataset(Dataset):
    def __init__(self, image_path, mask_path, npz_folder, val):
        dataset_type = 'validation' if val else 'training'
        self.image_path = image_path
        self.mask_path = mask_path
        self.npz_folder = npz_folder  # NPZ Folder Path

        npz_files_found = glob.glob(os.path.join(npz_folder, '*.npz'))
        print(f"Total {dataset_type} images found: {len(self.image_path)}")
        print(f"Total {dataset_type} masks found: {len(self.mask_path)}\n")
        print(f"Total {dataset_type} heatmaps found: {len(npz_files_found)}\n")

        if len(self.image_path) != len(self.mask_path):
            raise ValueError("Image and mask paths must have the same length")


    def __len__(self):
        return len(self.image_path) * 2

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_path)
        # 1. Load Image
        img_path_str = self.image_path[real_idx]
        img_bgr = cv2.imread(img_path_str)

        if img_bgr is None:  # Safety skip
            return self.__getitem__((real_idx + 1) % len(self))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(self.mask_path[real_idx], cv2.IMREAD_GRAYSCALE)
        mask_full = (mask_gray > 128).astype(np.uint8)

        # 2. Load Pre-Computed Targets (NPZ)
        # Logic: Convert "path/to/image.png" -> "path_image.npz" to match the converter script
        p = Path(img_path_str)
        clean_filename = f"{p.parent.name}_{p.stem.replace('_main', '')}.npz"
        npz_path = os.path.join(self.npz_folder, clean_filename)

        if os.path.exists(npz_path):
            # Load specific file
            data = np.load(npz_path)
            direction = data['direction']
            energy = data['energy']
        else:
            # Silent fallback for debugging
            direction = np.zeros((2, img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)
            energy = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)

        # 3. Deterministic 50% crop
        h_total, w_total, _ = img_rgb.shape
        crop_size = 512

        top = 0
        pass_number = idx // len(self.image_path)
        left = 0 if pass_number == 0 else (w_total - crop_size)

        img_crop = img_rgb[top:top + crop_size, left:left + crop_size, :]
        mask_crop = mask_full[top:top + crop_size, left:left + crop_size]
        energy_crop = energy[top:top + crop_size, left:left + crop_size]
        direction_crop = direction[:, top:top + crop_size, left:left + crop_size]

        # 4. ImageNet Normalization
        rgb_norm = img_crop.astype(np.float32) / 255.0
        rgb_norm = (rgb_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        mask_expanded = np.expand_dims(mask_crop, 2).astype(np.float32)
        rgb_gated = rgb_norm * mask_expanded
        input_np = np.concatenate([rgb_gated, mask_expanded], axis=2)

        return (
            torch.from_numpy(input_np).permute(2, 0, 1).float(),
            torch.from_numpy(direction_crop).float(),
            torch.from_numpy(energy_crop).float() # Continuous regression output
        )