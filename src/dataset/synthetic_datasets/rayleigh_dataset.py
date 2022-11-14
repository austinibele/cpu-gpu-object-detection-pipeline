from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

class RayleighDataset(Dataset):
    def __init__(self, num_samples, image_size=(32, 32), target_size=5, z_score_range=(5, 10), thickness=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.target_size = target_size
        self.z_score_range = z_score_range
        
        self.thickness = thickness
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image, target = self._generate_image_target()
        return image, target
        
    def _generate_image_target(self):
        bg = np.random.rayleigh(scale=1.0, size=self.image_size)
        x_min = np.random.randint(self.thickness, self.image_size[0] - self.target_size-self.thickness)
        y_min = np.random.randint(self.thickness, self.image_size[1] - self.target_size - self.thickness)
        box = [x_min, y_min, x_min + self.target_size, y_min + self.target_size]    
        bg[box[0]:box[2], box[1]:box[3]] = bg[box[0]:box[2], box[1]:box[3]] + self.new_object()
        snr = self._calculate_snr(bg, box)
        target = {"boxes": torch.tensor([box]), "labels": torch.zeros(1), "snr": [snr]}
        bg = torch.tensor(bg, dtype=torch.float32)
        return bg, target
    
    def _calculate_snr(self, bg, box):
        noise = np.copy(bg[box[0]-self.thickness:box[2]+self.thickness, box[1]-self.thickness:box[3]+self.thickness])
        noise[self.thickness:-self.thickness, self.thickness:-self.thickness] = 0
        noise_mean = noise[noise != 0].mean()
        
        object_max = bg[box[0]:box[2], box[1]:box[3]].max()
        snr = 10*np.log10(object_max/noise_mean)
        return snr
    
    def new_object(self):
        object = np.zeros(shape=(self.target_size, self.target_size))
        object[self.target_size//2, self.target_size//2] = np.ones(1)*30
        object = gaussian_filter(object, sigma=1)
        return object
    
