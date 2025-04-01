from paddleocr import PaddleOCR
from OCR_Models.BaseOCR import BaseOCR

class PaddleOCR(BaseOCR):
    def __init__(self, language="en", gpu=False, poppler_path=r"C:\\poppler\\Library\\bin"):
        super().__init__(language, gpu, poppler_path)
        # Initialize PaddleOCR with language and GPU settings
        self.ocr = PaddleOCR(use_angle_cls=True, lang=language, use_gpu=gpu)

    def ocr_image(self, file_path):
        # Convert PDF to images (assuming convert_pdf_to_img is implemented in BaseOCR)
        images = self.convert_pdf_to_img(file_path)
        
        text_results = []
        for img_path in images:
            # Run OCR on the image. Det and cls parameters are used for detection and angle classification.
            result = self.ocr.ocr(img_path, cls=True)  # Use cls for text orientation detection
            # Extract the text part of the OCR result
            text = [line[1][0] for line in result[0]]  # Get only the text part
            text_results.append(text)
        
        return text_results

    def ocr_image_with_conf(self, file_path):
        # Convert PDF to images (assuming convert_pdf_to_img is implemented in BaseOCR)
        images = self.convert_pdf_to_img(file_path)
        
        detailed_results = []
        for img_path in images:
            # Run OCR on the image. Det and cls parameters are used for detection and angle classification.
            result = self.ocr.ocr(img_path, cls=True)  # cls=True enables angle classification
            
            # Detailed results include text, box coordinates, and confidence score
            detailed_info = [
                {"text": line[1][0], "confidence": line[1][1], "box": line[0]} 
                for line in result[0]  # Extract text, confidence, and box coordinates
            ]
            detailed_results.append(detailed_info)
        
        return detailed_results
