from src.training.training_session.training_session_arg_parser import TrainingSessionArgParser
from src.model.model_loader import ModelLoader
from src.dataset.dataset_building.dataset_builder import DatasetBuilder
from torch.utils.data import DataLoader
from src.config.config_loader import ConfigLoader
from src.training.ptl_trainer import PtlTrainer
from src.dataset.utils.dataset_splitters import ordered_train_val_split
from src.model.lightning_modules.object_detection_module import ObjectDetectionModule
from src.dataset.utils.utils import retinanet_collate_fn

class TrainingSession:
    def __init__(self, args):
        self.args = args        
        self.load_config()
        self.load_model_to_module()
        self.load_data()
        
    def load_config(self):
        self.config = ConfigLoader.load(self.args.config_path)
        
    def load_model_to_module(self):
        model = ModelLoader.load(model_type=self.config.model_type)
        self.model = ObjectDetectionModule(model=model, lr=self.config.learning_rate)
   
    def load_data(self):
        dataset = DatasetBuilder.build_dataset(dataset_type=self.config.dataset_type)
        train_dataset, val_dataset = ordered_train_val_split(dataset, train_proportion=0.8)
        
        self.train_dataloader = DataLoader(train_dataset,
                                           collate_fn=retinanet_collate_fn,
                                           batch_size=self.config.batch_size,
                                           shuffle=True,
                                           num_workers=self.config.num_workers)
        
        self.val_dataloader = DataLoader(val_dataset,
                                        collate_fn=retinanet_collate_fn,
                                        batch_size=self.config.batch_size,
                                        shuffle=False,
                                        num_workers=self.config.num_workers)
    
    
    def train(self):
        trainer = PtlTrainer(model=self.model, train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader)
        trainer.train()
        
    
if __name__ == '__main__':
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.train()

