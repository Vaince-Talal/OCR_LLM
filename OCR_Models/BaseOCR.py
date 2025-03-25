from abc import ABC, abstractmethod


class BaseOCR(ABC):
    def __init__(self, language="en", gpu=False):
        self.language = language
        self.gpu = gpu

    @abstractmethod
    def ocr_image(self, image_path):
        pass

    @abstractmethod
    def ocr_image_with_conf(self, image_path):
        pass
