from src.dataset.synthetic_datasets.rayleigh_dataset import RayleighDataset


def test_rayleigh_dataset():
    dataset = RayleighDataset(num_samples=10, image_size=(32, 32), target_size=5, z_score_range=(5,10))
    image, target = dataset[0]
    assert image.shape == (32, 32)
    assert list(target.keys()) == ["boxes", "labels", "snr"]

    
if __name__ == '__main__':
    test_rayleigh_dataset()
    
    