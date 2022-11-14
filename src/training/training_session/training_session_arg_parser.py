from argparse import ArgumentParser

class TrainingSessionArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        
        self.add_argument(
            "--config_path",
            type=str,
            default="config/training/default.json",
            required=False,
            help="the config file for the training run"
        )


