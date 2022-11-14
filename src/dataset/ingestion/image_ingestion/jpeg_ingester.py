from torchvision.io import read_image
import torch

class JpegIngester:
    @classmethod
    def read(cls, image_path):
        return read_image(image_path).to(dtype=torch.float32)
