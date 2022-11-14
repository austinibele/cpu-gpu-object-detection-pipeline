from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

