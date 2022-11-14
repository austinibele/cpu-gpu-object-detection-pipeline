
def retinanet_collate_fn(batch):  
    return [b[0] for b in batch], [b[1] for b in batch]
