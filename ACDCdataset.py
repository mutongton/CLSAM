import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np


class ACDCDataset(Dataset):
    """
    适配所有ACDC数据 + 强制维度清理 + 读取前重写文件
    彻底解决 interpolate 维度报错问题
    """

    def __init__(self, root_dir, target_size=(256,256), num_frames=4):
        self.root_dir = root_dir
        self.target_size = target_size
        self.num_frames = num_frames
        self.data_pairs = self._load_acdc_data()

    def _load_acdc_data(self):
        # 加载ACDC官方训练集配对
        training_dir = os.path.join(self.root_dir, "training")
        pairs = []
        for patient in os.listdir(training_dir):
            p_path = os.path.join(training_dir, patient)
            if not os.path.isdir(p_path):
                continue
            files = os.listdir(p_path)
            imgs = sorted([f for f in files if f.endswith(".nii.gz") and "_gt" not in f and "_4d" not in f ])
            lbls = sorted([f for f in files if f.endswith("_gt.nii.gz")])
            for img, lbl in zip(imgs, lbls):
                pairs.append((
                    os.path.join(p_path, img),
                    os.path.join(p_path, lbl)
                ))
        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def _rewrite_and_clean(self, path):
        """
        1. 读取 → 重写 → 清理所有多余维度 → 保证输出为 3维 (D, H, W)
        【核心修复：强制squeeze去除所有单维度】
        """
        print("Processing:", path)
        itk_img = sitk.ReadImage(path)

        # 🔥 关键：压缩所有维度为 3维 (D, H, W)，解决所有维度报错
        arr = sitk.GetArrayFromImage(itk_img)
        arr = np.squeeze(arr)  # 强制去掉所有单维度，比如 [1,D,H,W] → [D,H,W]
        return arr

    def resize_3d_safe(self, data, is_label):
        x = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        target = (self.num_frames, self.target_size[0], self.target_size[1])

        if is_label:
            x = x.float()  # 🔥 必须
            x = F.interpolate(x, size=target, mode="nearest")
            x = x.long()  # 🔥 再转回类别
        else:
            x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)

        return x.squeeze().numpy()

    def __getitem__(self, idx):
        img_path, lbl_path = self.data_pairs[idx]

        # 1. 读取+重写+清理维度 (强制3维)
        image = self._rewrite_and_clean(img_path).astype(np.float32)
        label = self._rewrite_and_clean(lbl_path).astype(np.int64)

        # 2. 标准化
        image = (image - image.mean()) / (image.std() + 1e-8)

        # 3. 安全插值
        image = self.resize_3d_safe(image, is_label=False)
        label = self.resize_3d_safe(label, is_label=True)

        # 4. 转为模型输入 [C, T, H, W]
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return image, label