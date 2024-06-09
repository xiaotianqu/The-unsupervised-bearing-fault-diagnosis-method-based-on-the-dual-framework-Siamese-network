import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import random
height, width = 227, 227

class SiameseDataset(Dataset):
    def __init__(self, folder_path,  digit_index=0):
        self.folder_path = folder_path
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        self.digit_index = digit_index  # 参数来指定要提取的数字的位置

        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.folder_path, image_file)
        label = int(image_file.split('_')[self.digit_index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            other_index = random.choice([i for i in range(len(self.image_files)) if
                                         int(self.image_files[i].split('_')[self.digit_index]) == label])
        else:
            other_index = random.choice([i for i in range(len(self.image_files)) if
                                         int(self.image_files[i].split('_')[self.digit_index]) != label])

        other_image_file = self.image_files[other_index]
        other_image_path = os.path.join(self.folder_path, other_image_file)
        other_label = int(other_image_file.split('_')[self.digit_index])

        other_image = Image.open(other_image_path).convert("RGB")
        other_image = self.transform(other_image)

        return image, other_image, torch.tensor(int(label != other_label), dtype=torch.float32)  # 返回两个图像和标签
class TestSiameseDatasetA(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image
class TestSiameseDatasetB(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image