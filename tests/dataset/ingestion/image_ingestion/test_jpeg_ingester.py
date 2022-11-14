from src.dataset.ingestion.image_ingestion.jpeg_ingester import JpegIngester
import torch

def test_read():
    image_path = "tests/dataset/ingestion/image_ingestion/jpg_fixture.jpg"
    t = JpegIngester.read(image_path)
    assert t.shape == (3, 831, 859)
    assert t.dtype == torch.uint8
    
if __name__ == "__main__":
    test_read()
