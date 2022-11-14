from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

class PtlTrainer:
    def __init__(self, model, train_dataloader, val_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
    def train(self):
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=100,
            gpus=1,
            logger=None,
            callbacks=[],
            deterministic=True,
            accelerator="auto",
            check_val_every_n_epoch=1
        )
        trainer.fit(model=self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
    
    
