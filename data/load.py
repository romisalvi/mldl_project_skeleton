import os
import shutil
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class TinyImageNetDataLoader:
    def __init__(self, data_dir='tiny-imagenet/tiny-imagenet-200', batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Resize images to match model input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_dataset = ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=self.transform)
        self.val_dataset = ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
    def get_loaders(self):
        """Returns the training and validation DataLoaders."""
        return self.train_loader, self.val_loader
