import pytesseract
from pytesseract import Output
from BaseOCR import BaseOCR

class TesseractOCR(BaseOCR):
    def __init__(self, language="eng", gpu=False):
        super().__init__(language, gpu)

    def ocr_image(self, image_path):
        text = pytesseract.image_to_string(image_path, lang=self.language)
        return text

    def ocr_image_with_conf(self, image_path):
        data = pytesseract.image_to_data(image_path, lang=self.language, output_type=Output.DICT)
        results = []
        for i, text in enumerate(data['text']):
            conf = int(data['conf'][i])
            if text.strip():
                results.append({"text": text, "confidence": conf})
        return results
