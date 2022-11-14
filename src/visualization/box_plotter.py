import matplotlib.patches as patches
import torch
import matplotlib.pyplot as plt

class BoxPlotter:
    def __init__(self):
        pass

    def draw(self, image, target):
        image = self._coerce_to_have_channel_dim(image)
        image = torch.moveaxis(image, 0, 2)
        
        fig, ax = plt.subplots()
        ax = self._add_rectangles(ax=ax, target=target)

        ax.imshow(image)
        
    def _add_rectangles(self, ax, target):
        for box in target["boxes"]:
            rect = self._get_rectangle(box)
            ax.add_patch(rect)
        return ax
    
    def _get_rectangle(self, box):
        return patches.Rectangle((box[1]-0.5, box[0]-0.5), box[3]-box[1], box[2]-box[0], linewidth=1, edgecolor='r', facecolor='none')
    
    def _coerce_to_have_channel_dim(self, image):
        if len(image.shape) == 2:
            image = image.unsqueeze(dim=0)
        elif len(image.shape) == 4:
            raise ValueError(f"The image passed has dimensions (B, C, H, W). Please pass individual images in form (C, H, W), not batches.")
        return image 
