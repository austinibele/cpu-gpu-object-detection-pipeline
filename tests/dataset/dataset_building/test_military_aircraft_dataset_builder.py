from src.dataset.dataset_building.military_aircraft.military_aircraft_dataset_builder import MilitaryAircraftDatasetBuilder

def test_build_dataset():
    builder = MilitaryAircraftDatasetBuilder()
    dataset = builder.build_dataset()
    img, tar = dataset[0]
    assert img.shape == (3, 831, 859)
    assert tar["names"] == ['A2', 'A2', 'A10']
    
if __name__ == '__main__':
    test_build_dataset()

