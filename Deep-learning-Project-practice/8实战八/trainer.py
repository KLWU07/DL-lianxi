import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from voc_dataset import VOCDataset
from transforms import get_transform
from model_builder import get_model


# 定义显式的collate函数
def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')

    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        progress_bar.set_postfix({'total_loss': losses.item()})


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012'

    train_dataset = VOCDataset(root_dir, split='train', transforms=get_transform(train=True))
    val_dataset = VOCDataset(root_dir, split='val', transforms=get_transform(train=False))

    # 使用定义好的collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=collate_fn  # 替换lambda
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=collate_fn  # 替换lambda
    )

    num_classes = 21
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        torch.save(model.state_dict(), f'faster_rcnn_voc_epoch_{epoch}.pth')

    print("训练完成!")


if __name__ == "__main__":
    main()