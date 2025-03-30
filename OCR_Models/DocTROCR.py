from OCR_Models.BaseOCR import BaseOCR
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class DocTROCR(BaseOCR):
    def __init__(self, language="en", gpu=False, poppler_path=r"C:\\poppler\\Library\\bin"):
        super().__init__(language, gpu, poppler_path)
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.model = ocr_predictor(pretrained=True)

    def ocr_image(self, file_path):
        text_results = []
        doc = DocumentFile.from_pdf(file_path)
        result = self.model(doc)
        print(result)
        extracted_text = "\n".join([word.value for block in result.pages[0].blocks for line in block.lines for word in line.words])
        text_results.append(extracted_text)
        return "\n".join(text_results)

    def ocr_image_with_conf(self, file_path):
        images = self.convert_pdf_to_img(file_path)
        detailed_results = []
        for img_path in images:
            doc = DocumentFile.from_images(img_path)
            result = self.model(doc)
            for block in result.pages[0].blocks:
                for line in block.lines:
                    for word in line.words:
                        detailed_results.append({
                            "text": word.value,
                            "confidence": word.confidence
                        })
        return detailed_results