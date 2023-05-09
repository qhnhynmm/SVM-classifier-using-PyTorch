import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
class LoadData:
    def __init__(self, config):
        self.image_H = config.image_H
        self.image_W = config.image_W
        self.batch_size = config.batch_size
        
    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

    def load_data(self, data_path):
        transform = transforms.Compose([
            transforms.Resize((self.image_H, self.image_W)),
            transforms.RandomCrop(size=(self.image_H, self.image_W), padding=4),       
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        return dataloader
        
    def load_test_data(self, data_path):
        transform = transforms.Compose([
            transforms.Resize((self.image_H, self.image_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        return test_dataloader