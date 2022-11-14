from src.training.training_session.training_session_arg_parser import TrainingSessionArgParser
from src.model.model_loader import ModelLoader
from src.dataset.dataset_builder.dataset_builder import DatasetBuilder
from torch.utils.data import DataLoader
from src.config.config_loader import ConfigLoader
from src.training.ptl_trainer import PtlTrainer

class TrainingSession:
    def __init__(self, args):
        self.args = args        
        self.load_config()
        self.load_model()
        self.load_data()
        
    def load_config(self):
        self.config = ConfigLoader.load(self.args.config_path)
        
    def load_model(self):
        self.model = ModelLoader.load(model_type=self.config.model_type)
   
    def load_data(self):
        train_dataset = DatasetBuilder.build_train_dataset()
        val_dataset = DatasetBuilder.build_val_dataset()
        
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.config.batch_size,
                                           shuffle=True,
                                           num_workers=self.config.num_workers)
        self.val_dataloader = DataLoader(val_dataset,
                                           batch_size=self.config.batch_size,
                                           shuffle=True,
                                           num_workers=self.config.num_workers)
    
    def train(self):
        trainer = PtlTrainer(model=self.model, train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader)
        trainer.train()
        
    
if __name__ == '__main__':
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.train()

