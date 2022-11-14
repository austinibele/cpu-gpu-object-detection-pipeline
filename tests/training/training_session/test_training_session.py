from collections import namedtuple
from src.training.training_session.training_session import TrainingSession

Args = namedtuple("Args", ["model_type"])
args = Args("retinanet")

def test_init():
    session = TrainingSession(args=args) 
    assert session.args.model_type == "retinanet"
    
if __name__ == '__main__':
    test_init()
