import os
from abc import ABC, abstractmethod

from pdf2image import convert_from_path


class BaseOCR(ABC):
    def __init__(self, language="en", gpu=False, poppler_path=r"C:\\poppler\\Library\\bin"):
        self.language = language
        self.gpu = gpu
        self.poppler_path = poppler_path

    def convert_pdf_to_img(self, file_path, output_folder="images"):
        if file_path.lower().endswith('.pdf'):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            images = convert_from_path(
                file_path, output_folder=output_folder, poppler_path=self.poppler_path
            )
            image_paths = []
            for i, image in enumerate(images):
                output_image_path = os.path.join(output_folder, f"page_{i + 1}.png")
                image.save(output_image_path, "PNG")
                image_paths.append(output_image_path)
            return image_paths
        else:
            # It's already an image, return directly in list form for consistency
            return [file_path]
    @abstractmethod
    def ocr_image(self, image_path):
        pass

    @abstractmethod
    def ocr_image_with_conf(self, image_path):
        pass
