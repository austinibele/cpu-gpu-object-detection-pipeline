import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class LitModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def training_epoch_end(self, training_step_outputs):
        save_path = "TODO" # TODO
        torch.save(self.model.state_dict(), save_path)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
 
    def validation_epoch_end(self, validation_step_outputs):
        # compute validation metrics
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def load_weights(self):
        # load weights if training from checkpoint
        raise NotImplementedError
    