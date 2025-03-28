import os
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

from OCR_Models.BaseOCR import BaseOCR


class SuryaOCR(BaseOCR):
    def __init__(self, language="en", gpu=False, poppler_path=r"C:\\poppler\\Library\\bin"):
        super().__init__(language, gpu, poppler_path)
        self.langs = [language] if language else None
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

    def ocr_image(self, file_path):
        images = self.convert_pdf_to_img(file_path)
        text_results = []

        for img_path in images:
            image = Image.open(img_path)
            predictions = self.recognition_predictor([image], [self.langs], self.detection_predictor)

            for result in predictions:
                for line in result.text_lines:
                    text_results.append(line.text)

        return "\n".join(text_results)

    def ocr_image_with_conf(self, file_path):
        images = self.convert_pdf_to_img(file_path)
        detailed_results = []

        for img_path in images:
            image = Image.open(img_path)
            predictions = self.recognition_predictor([image], [self.langs], self.detection_predictor)

            for result in predictions:
                for line in result.text_lines:
                    detailed_results.append({
                        "text": line.text,
                        "confidence": line.confidence
                    })

        return detailed_results
