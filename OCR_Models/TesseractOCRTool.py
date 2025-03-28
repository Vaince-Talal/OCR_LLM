import pytesseract
from pytesseract import Output

from OCR_Models.BaseOCR import BaseOCR


class TesseractOCR(BaseOCR):
    def __init__(self, language="eng", gpu=False, poppler_path=r"C:\\poppler\\Library\\bin"):
        super().__init__(language, gpu, poppler_path)

    def ocr_image(self, file_path):
        images = self.convert_pdf_to_img(file_path)
        text_results = []
        for img in images:
            text = pytesseract.image_to_string(img, lang=self.language)
            text_results.append(text)
        return "\n".join(text_results)

    def ocr_image_with_conf(self, file_path):
        images = self.convert_pdf_to_img(file_path)
        detailed_results = []
        for img in images:
            data = pytesseract.image_to_data(img, lang=self.language, output_type=Output.DICT)
            for i, text in enumerate(data['text']):
                conf = int(data['conf'][i])
                if text.strip():
                    detailed_results.append({"text": text, "confidence": conf})
        return detailed_results
