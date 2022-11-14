from src.dataset.detection_dataset import DetectionDataset
from src.dataset.ingestion.file_finder import FileFinder
from src.dataset.ingestion.image_ingestion.jpeg_ingester import JpegIngester
from src.dataset.ingestion.target_ingestion.xml_parser import XmlParser
from tqdm import tqdm

class MilitaryAircraftDatasetBuilder:
    def build_dataset(self, **kwargs):
        image_files = FileFinder.find_files("data/military_aircraft/JPEGImages", ".jpg", limit=100)
        annotation_files = FileFinder.find_files("data/military_aircraft/Annotations/horizontal_bounding_boxes", ".xml", limit=100)
        
        print("Loading Data...")
        images = []
        targets = []
        for i, image_path in enumerate(tqdm(image_files)):
            images.append(JpegIngester.read(image_path))
            targets.append(XmlParser.parse_bbox_annotation(annotation_files[i]))
        
        assert len(images) == len(targets)
        return DetectionDataset(images=images, targets=targets)
