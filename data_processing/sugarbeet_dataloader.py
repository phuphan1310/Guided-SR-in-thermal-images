# data/sugarbeet_dataloader.py
from torchvision import transforms
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset
import torch
import random
import re

class SugarBeet_Dataset(Dataset):
    def __init__(self, root_dir, scale=4, transform=None, train=True, max_samples=None):
        """
        Args:
            root_dir: đường dẫn đến thư mục gốc (chứa train/ và val/)
            scale: scale factor (4)
            transform: transforms
            train: True cho train, False cho val
            max_samples: số lượng mẫu tối đa
        """
        self.transform = transform
        self.scale = scale
        self.max_samples = max_samples
        
        # Kích thước patch cố định - ĐÃ SỬA
        self.gt_h, self.gt_w = 128, 160  # GT/RGB patch: 128x160
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR patch: 32x40
        
        # Kích thước ảnh gốc
        self.rgb_gt_h, self.rgb_gt_w = 964, 1296  # RGB và GT: 1296x964
        self.depth_h, self.depth_w = 241, 324     # Depth: 324x241 (đã là LR)
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        self.train = train
        
        # Kiểm tra thư mục
        print(f"\n=== Checking SugarBeet {split} directories ===")
        print(f"RGB dir: {self.rgb_dir} - exists: {os.path.exists(self.rgb_dir)}")
        print(f"Depth dir: {self.depth_dir} - exists: {os.path.exists(self.depth_dir)}")
        print(f"GT dir: {self.gt_dir} - exists: {os.path.exists(self.gt_dir)}")
        
        # Lấy danh sách file
        rgb_files = glob(os.path.join(self.rgb_dir, '*.png')) + glob(os.path.join(self.rgb_dir, '*.jpg'))
        depth_files = glob(os.path.join(self.depth_dir, '*.png')) + glob(os.path.join(self.depth_dir, '*.jpg'))
        gt_files = glob(os.path.join(self.gt_dir, '*.png')) + glob(os.path.join(self.gt_dir, '*.jpg'))
        
        print(f"\nFound files:")
        print(f"  RGB: {len(rgb_files)} files")
        print(f"  Depth: {len(depth_files)} files")
        print(f"  GT: {len(gt_files)} files")
        
        # Hàm trích xuất base name (bỏ qua phần _rgb, _nir, _gt)
        def extract_base_name(filename):
            basename = os.path.basename(filename)
            # Pattern: bonirob_2016-04-20-15-37-25_0_nir_00000
            match = re.match(r'(.*?)_(rgb|nir|gt)_(.*)', basename)
            if match:
                return f"{match.group(1)}_{match.group(3)}"
            return os.path.splitext(basename)[0]
        
        # Tạo dictionary với key là base name chung
        rgb_dict = {}
        for f in rgb_files:
            key = extract_base_name(f)
            if key not in rgb_dict:
                rgb_dict[key] = f
        
        depth_dict = {}
        for f in depth_files:
            key = extract_base_name(f)
            if key not in depth_dict:
                depth_dict[key] = f
        
        gt_dict = {}
        for f in gt_files:
            key = extract_base_name(f)
            if key not in gt_dict:
                gt_dict[key] = f
        
        # Tìm tên chung
        common_names = set(rgb_dict.keys()) & set(depth_dict.keys()) & set(gt_dict.keys())
        common_names = sorted(list(common_names))
        
        print(f"\nCommon base names: {len(common_names)}")
        if len(common_names) > 0:
            print(f"First 5: {common_names[:5]}")
        
        # Lấy đường dẫn file tương ứng
        self.rgb_files = [rgb_dict[name] for name in common_names]
        self.depth_files = [depth_dict[name] for name in common_names]
        self.gt_files = [gt_dict[name] for name in common_names]
        
        # Giới hạn số lượng samples
        if max_samples is not None and len(self.rgb_files) > max_samples:
            if train:
                indices = random.sample(range(len(self.rgb_files)), max_samples)
                self.rgb_files = [self.rgb_files[i] for i in indices]
                self.depth_files = [self.depth_files[i] for i in indices]
                self.gt_files = [self.gt_files[i] for i in indices]
            else:
                self.rgb_files = self.rgb_files[:max_samples]
                self.depth_files = self.depth_files[:max_samples]
                self.gt_files = self.gt_files[:max_samples]
        
        # Tính các vị trí top, left hợp lệ cho RGB và GT (chia hết cho scale)
        self.valid_tops = list(range(0, self.rgb_gt_h - self.gt_h + 1, self.scale))
        self.valid_lefts = list(range(0, self.rgb_gt_w - self.gt_w + 1, self.scale))
        
        # Tính các vị trí top, left hợp lệ cho Depth
        self.depth_valid_tops = list(range(0, self.depth_h - self.lr_h + 1, 1))
        self.depth_valid_lefts = list(range(0, self.depth_w - self.lr_w + 1, 1))
        
        print(f"\n=== SugarBeet {split} set ===")
        print(f"Total valid files: {len(self.rgb_files)}")
        print(f"RGB/GT size: {self.rgb_gt_h}x{self.rgb_gt_w}")
        print(f"Depth size: {self.depth_h}x{self.depth_w}")
        print(f"GT/RGB patch size: {self.gt_h}x{self.gt_w}")
        print(f"LR patch size: {self.lr_h}x{self.lr_w}")
        print(f"Valid top positions (RGB/GT): {len(self.valid_tops)}")
        print(f"Valid left positions (RGB/GT): {len(self.valid_lefts)}")
        print(f"Valid top positions (Depth): {len(self.depth_valid_tops)}")
        print(f"Valid left positions (Depth): {len(self.depth_valid_lefts)}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Đọc ảnh
        rgb = cv2.imread(self.rgb_files[idx])
        if rgb is None:
            raise ValueError(f"Không thể đọc RGB file: {self.rgb_files[idx]}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Không thể đọc Depth file: {self.depth_files[idx]}")
        
        gt = cv2.imread(self.gt_files[idx], cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Không thể đọc GT file: {self.gt_files[idx]}")
        
        # Kiểm tra kích thước RGB và GT
        h_gt, w_gt = gt.shape
        h_rgb, w_rgb = rgb.shape[:2]
        
        if h_gt != self.rgb_gt_h or w_gt != self.rgb_gt_w:
            print(f"Warning: GT có kích thước {h_gt}x{w_gt}, resize về {self.rgb_gt_h}x{self.rgb_gt_w}")
            gt = cv2.resize(gt, (self.rgb_gt_w, self.rgb_gt_h))
        
        if h_rgb != self.rgb_gt_h or w_rgb != self.rgb_gt_w:
            print(f"Warning: RGB có kích thước {h_rgb}x{w_rgb}, resize về {self.rgb_gt_h}x{self.rgb_gt_w}")
            rgb = cv2.resize(rgb, (self.rgb_gt_w, self.rgb_gt_h))
        
        # Kiểm tra kích thước Depth
        h_depth, w_depth = depth.shape
        if h_depth != self.depth_h or w_depth != self.depth_w:
            print(f"Warning: Depth có kích thước {h_depth}x{w_depth}, resize về {self.depth_h}x{self.depth_w}")
            depth = cv2.resize(depth, (self.depth_w, self.depth_h))
        
        # Chọn vị trí patch cho RGB và GT (phải chia hết cho scale)
        if self.train:
            top = random.choice(self.valid_tops)
            left = random.choice(self.valid_lefts)
            
            # Chọn vị trí tương ứng cho Depth (tọa độ tương ứng với HR)
            depth_top = top // self.scale
            depth_left = left // self.scale
            
            # Đảm bảo depth_top, depth_left nằm trong khoảng cho phép
            depth_top = min(depth_top, self.depth_h - self.lr_h)
            depth_left = min(depth_left, self.depth_w - self.lr_w)
        else:
            # Center crop cho validation
            top = self.valid_tops[len(self.valid_tops) // 2]
            left = self.valid_lefts[len(self.valid_lefts) // 2]
            depth_top = top // self.scale
            depth_left = left // self.scale
        
        # Cắt patch trên GT và RGB
        gt_patch = gt[top:top+self.gt_h, left:left+self.gt_w]
        rgb_patch = rgb[top:top+self.gt_h, left:left+self.gt_w, :]
        
        # Cắt patch trên Depth
        lr_patch = depth[depth_top:depth_top+self.lr_h, depth_left:depth_left+self.lr_w]
        
        # Kiểm tra kích thước patch
        assert gt_patch.shape == (self.gt_h, self.gt_w), f"GT patch size sai: {gt_patch.shape}"
        assert rgb_patch.shape[:2] == (self.gt_h, self.gt_w), f"RGB patch size sai: {rgb_patch.shape}"
        assert lr_patch.shape == (self.lr_h, self.lr_w), f"LR patch size sai: {lr_patch.shape}"
        
        # Normalize
        rgb_patch = rgb_patch.astype(np.float32) / 255.0
        lr_patch = lr_patch.astype(np.float32) / 255.0
        gt_patch = gt_patch.astype(np.float32) / 255.0
        
        # Transform
        if self.transform:
            rgb_patch = self.transform(rgb_patch)
            lr_patch = torch.from_numpy(lr_patch).unsqueeze(0).float()
            gt_patch = torch.from_numpy(gt_patch).unsqueeze(0).float()
        else:
            rgb_patch = torch.from_numpy(np.transpose(rgb_patch, (2, 0, 1))).float()
            lr_patch = torch.from_numpy(lr_patch).unsqueeze(0).float()
            gt_patch = torch.from_numpy(gt_patch).unsqueeze(0).float()

        return {
            'guidance': rgb_patch,  # [3, 128, 160]
            'lr': lr_patch,         # [1, 32, 40]
            'gt': gt_patch           # [1, 128, 160]
        }