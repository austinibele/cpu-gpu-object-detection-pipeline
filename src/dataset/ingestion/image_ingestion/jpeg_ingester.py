from torchvision.io import read_image

class JpegIngester:
    @classmethod
    def read(cls, image_path):
        return read_image(image_path)
