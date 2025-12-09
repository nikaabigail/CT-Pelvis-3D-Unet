import os
import torch

from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import MapTransform


class PrintShape(MapTransform):
    def __call__(self, data):
        print("IMAGE SHAPE:", data["image"].shape)
        print("LABEL SHAPE:", data["label"].shape)
        return data

root = r"D:\Dic\MONAI_dataset"

images_dir = os.path.join(root, "images")
labels_dir = os.path.join(root, "labels")

images = sorted([
    os.path.join(images_dir, f)
    for f in os.listdir(images_dir)
    if f.endswith(".nii.gz")
])
labels = sorted([
    os.path.join(labels_dir, f)
    for f in os.listdir(labels_dir)
    if f.endswith(".nii.gz")
])

import nibabel as nib

assert len(images) == len(labels), "Different number labels and images"

data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
print(f"Case total: {len(data_dicts)}")

val_num = 6
train_files = data_dicts[:-val_num]
val_files = data_dicts[-val_num:]


print(f"Train: {len(train_files)}, Val: {len(val_files)}")

patch_size = (128, 128, 128)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    PrintShape(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=3000,
        b_min=0.0, b_max=1.0,
        clip=True,
    ),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=patch_size,
        pos=1, neg=1,
        num_samples=2,
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=3000,
        b_min=0.0, b_max=1.0,
        clip=True,
    ),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


for i, val_data in enumerate(val_loader):
    print("VAL INDEX:", i)
    print("val_data type:", type(val_data))
    print("val_data keys:", val_data.keys() if isinstance(val_data, dict) else "NOT DICT")
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

num_classes = 9

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=True,
)
dice_metric = DiceMetric(
    include_background=True,
    reduction="mean",
    get_not_nans=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

max_epochs = 50
val_interval = 1
best_metric = -1
best_metric_epoch = -1
model_path = "best_unet.pth"

for epoch in range(1, max_epochs + 1):
    print(f"\n=== Epoch {epoch}/{max_epochs} ===")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if step % 10 == 0:
            print(f"  step {step}, loss = {loss.item():.4f}")

    epoch_loss /= step
    print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                val_outputs = sliding_window_inference(
                    val_images,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                )
                val_outputs = torch.softmax(val_outputs, dim=1)
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels)
                dice_metric(y_pred=val_outputs_list, y=val_labels_list)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Validation mean Dice: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f"New best model (epoch {epoch}), saved to {model_path}")

print(f"\nЛучший Dice = {best_metric:.4f} на эпохе {best_metric_epoch}")
