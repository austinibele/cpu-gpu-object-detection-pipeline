from src.dataset.ingestion.xml_parser import XmlParser
import torch

def test_parse():
    xml_path = "tests/dataset/ingestion/xml_annotation_fixture.xml"
    target = XmlParser.parse_bbox_annotation(xml_path=xml_path)
    assert target["boxes"].sum() == torch.tensor([[485., 427., 554., 500.],
        [694., 487., 770., 562.],
        [ 58., 205., 134., 285.]]).sum()
    assert target["names"] == ['A2', 'A2', 'A10']
    assert len(target["labels"]) == 3
    assert target["labels"][0] == torch.tensor(2)

if __name__ == "__main__":
    test_parse()
    
