from src.config.config_loader import ConfigLoader
from src.config.config import Config

def test_load():
    config_path = "config/training/default.json"
    data = ConfigLoader.load(config_path)
    assert isinstance(data, Config)
    
if __name__ == '__main__':
    test_load()

