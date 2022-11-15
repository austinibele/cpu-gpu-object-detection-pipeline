import json
from src.config.config import Config

class ConfigLoader:
    @classmethod
    def load(cls, config_path):
        with open(config_path) as f:
            data = json.load(f)
        return Config(data)

    @classmethod
    def load_to_dict(cls, config_path):
        with open(config_path) as f:
            data = json.load(f)
        return data
