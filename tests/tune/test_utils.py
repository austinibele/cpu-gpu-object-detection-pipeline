from src.tune import utils

def test_exec_import_from_path():
    config_path = "config/tune/default.py"
    config = utils.import_config_from_path(config_path=config_path)
    assert isinstance(config, dict)
    
if __name__ == "__main__":
    test_exec_import_from_path()
