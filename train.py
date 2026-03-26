import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# ========== 导入自定义模块 ==========
from loss import CombinedLoss
from ACDCdataset import ACDCDataset
from validate import validate
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ========== 超参数配置（可自由修改） ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 50          # 训练轮次，自由设置
LEARNING_RATE = 1e-4
NUM_CLASSES = 4
INPUT_CHANNELS = 1
NUM_FRAMES = 4
IMAGE_SIZE = 256
VAL_SPLIT = 0.1      # 10% 数据做验证

# ========== 路径配置（必须修改） ==========
DATASET_ROOT = "ACDC"
SAVE_PATH = "./best_weights"
os.makedirs(SAVE_PATH, exist_ok=True)

# ========== 加载数据（训练集+验证集） ==========
if __name__ == "__main__":
    # 1. 导入你的 SAM 适配器模型
    from sam.sam_model_2024_acdc_patch1024_tqreshape import SAMAdapter_2024_ACDC_Patch1024_TQReshape

    # 2. 初始化完整数据集
    full_dataset = ACDCDataset(
        root_dir=DATASET_ROOT,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        num_frames=NUM_FRAMES
    )

    # 3. 划分训练集 / 验证集
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5. 初始化模型 / 损失 / 优化器
    model = SAMAdapter_2024_ACDC_Patch1024_TQReshape(
        input_channels=INPUT_CHANNELS,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 512, 512],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=3,
        strides=1,
        n_conv_per_stage=2,
        num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=2,
        frames=NUM_FRAMES,
        device=DEVICE
    ).to(DEVICE)

    criterion = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # ========== 训练最优值初始化 ==========
    best_val_loss = float('inf')

    # ========== 训练 + 验证 主循环 ==========
    print(f"训练启动 | 设备：{DEVICE} | 训练集：{train_size} | 验证集：{val_size}")
    for epoch in range(1, EPOCHS + 1):
        # --------------------- 训练一轮 ---------------------
        model.train()
        # 放在模型初始化之后
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结参数: {frozen_params:,}")
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast():  # 🔥 开启混合精度
                outputs = model(images)
                loss_total, loss_dice, loss_iou = criterion(outputs, labels)

            scaler.scale(loss_total).backward()  # 🔥 防止梯度下溢
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss_total.item()
            pbar.set_postfix({
                "Total": f"{loss_total.item():.3f}",
                "Dice": f"{loss_dice.item():.3f}",
                "IoU": f"{loss_iou.item():.3f}"
            })

        avg_train_loss = total_train_loss / len(train_loader)

        # --------------------- 验证一轮 ---------------------
        avg_val_loss = validate(model, val_loader, criterion, DEVICE)

        # --------------------- 打印结果 ---------------------
        print(f"\nEpoch {epoch} 完成")
        print(f"训练平均损失: {avg_train_loss:.4f} | 验证平均损失: {avg_val_loss:.4f}\n")

        # --------------------- 保存最优权重 ---------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_model.pth"))
            print(f"✅ 已保存最优模型 | 最优验证损失: {best_val_loss:.4f}")