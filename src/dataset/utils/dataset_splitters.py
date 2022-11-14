from torch.utils.data import Subset

def ordered_train_val_split(dataset, train_proportion=0.8):
    train_indices = range(0, int(len(dataset)*train_proportion))
    val_indices = range(int(len(dataset)*train_proportion) + 1, len(dataset))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
    