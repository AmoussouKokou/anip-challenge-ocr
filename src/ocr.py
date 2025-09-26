
from paddleocr import PaddleOCR

LANGS_BY_COUNTRY = {"es":["es","en"], "ee":["en"], "ru":["ru","en"], "az":["en"]}

class OCRWrapper:
    def __init__(self, country="es"):
        self.country = country
        # Note: 'multilang' suppose que les modèles multilingues sont installés
        self.ocr = PaddleOCR(use_angle_cls=True, lang='multilang', show_log=False)

    def run(self, img):
        res = self.ocr.ocr(img, cls=True)
        out=[]
        if res and len(res)>0:
            for line in res[0]:
                box = line[0]; text=line[1][0]; conf=line[1][1]
                out.append({"box":box, "text":text, "conf":float(conf)})
        return out
