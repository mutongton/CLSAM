import torch
from tqdm import tqdm

def validate(model, val_loader, criterion, device):
    """
    验证一轮
    :return: 平均验证总损失
    """
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss_total, _, _ = criterion(outputs, labels)

            total_val_loss += loss_total.item()
            pbar.set_postfix({"Val Loss": f"{loss_total.item():.3f}"})

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss