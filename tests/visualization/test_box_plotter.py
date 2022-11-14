from src.visualization.box_plotter import BoxPlotter
from src.dataset.synthetic_datasets.rayleigh_dataset import RayleighDataset

dataset = RayleighDataset(num_samples=1)
image, target = dataset[0]

def test_draw():
    plotter = BoxPlotter()
    plotter.draw(image, target)
    

if __name__ == '__main__':
    test_draw()
