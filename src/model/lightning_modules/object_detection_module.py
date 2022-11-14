import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class ObjectDetectionModule(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        
        self.metric = MeanAveragePrecision()
        self.epoch_counter = -1

    def training_step(self, batch, batch_idx):
        images, targets = batch
        losses = self.model(images, targets)
        classification_loss = losses["classification"]
        regression_loss = losses["bbox_regression"]
        loss = classification_loss + regression_loss
        return loss

    def training_epoch_end(self, training_step_outputs):
        save_path = "TODO" # TODO
        torch.save(self.model.state_dict(), save_path)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)
        _ = self.metric(predictions, targets)
 
    def validation_epoch_end(self, validation_step_outputs):
        acc = self.metric.compute()
        print(f"Accuracy on epoch {self.epoch_counter} = {acc}")
        self.metric.reset()
        self.epoch_counter += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def load_weights(self):
        # load weights if training from checkpoint
        raise NotImplementedError
    