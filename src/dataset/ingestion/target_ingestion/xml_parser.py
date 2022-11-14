import torch
import xml.etree.ElementTree as ET
from src.dataset.ingestion.utils import remove_alpha

class XmlParser:
    @classmethod
    def parse_bbox_annotation(cls, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        names = []
        boxes = []
        for child in root:
            if child.tag == "object":
                for subchild in child:
                    if subchild.tag == "name":
                        name = subchild.text
                        names.append(name)
                    if subchild.tag == "bndbox":
                        box = [None]*4
                        for grandchild in subchild:
                            if grandchild.tag == "xmin":
                                box[0] = int(grandchild.text)
                            elif grandchild.tag == "ymin":
                                box[1] = int(grandchild.text)
                            elif grandchild.tag == "xmax":
                                box[2] = int(grandchild.text)
                            elif grandchild.tag == "ymax":
                                box[3] = int(grandchild.text)
                        boxes.append(box)
        
        labels = []
        for name in names:
            labels.append(remove_alpha(name))

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["names"] = names
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        return target
