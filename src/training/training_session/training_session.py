from src.training.training_session.training_session_arg_parser import TrainingSessionArgParser


class TrainingSession:
    def __init__(self, args):
        self.args = args        
    
    
if __name__ == '__main__':
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.train()

