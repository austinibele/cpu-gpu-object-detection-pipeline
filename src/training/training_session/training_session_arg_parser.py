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
        self.add_argument(
            "--tune",
            type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
            default="config/training/default.json",
            required=False,
            help="the config file for the training run"
        )
        self.add_argument(
            "--tune_config_path",
            type=str,
            default="config/tune/default.py",
            required=False,
            help="the config file for the training run"
        )

