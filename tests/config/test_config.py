from src.config.config import Config

def test_config():
    data = {"key": "value"}
    config = Config(data)
    assert config.key == "value"

if __name__ == "__main__":
    test_config()
