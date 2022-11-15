import importlib

def import_config_from_path(config_path):
    module_path = config_path.replace("/", ".").replace(".py", "")
    module = importlib.import_module(module_path)
    return module.get_config()
