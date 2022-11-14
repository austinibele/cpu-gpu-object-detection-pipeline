from argparse import ArgumentParser

class TrainingSessionArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        
        self.add_argument(
            "--model_type",
            type=str,
            default="retinanet",
            required=False,
            help="the model type to use"
        )


