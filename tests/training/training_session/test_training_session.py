from collections import namedtuple
from src.training.training_session.training_session import TrainingSession

Args = namedtuple("Args", ["config_path"])
args = Args("config/training/default.json")

def test_init():
    session = TrainingSession(args=args) 
    assert session.args.config_path == "config/training/default.json"
    
if __name__ == '__main__':
    test_init()
