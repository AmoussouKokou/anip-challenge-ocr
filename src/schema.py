
import os, json, numpy as np

class SchemaManager:
    def __init__(self):
        self.templates = {}  # {country: {"template_image": ndarray or None, "fields":[{"key":..., "bbox":[x,y,w,h], "regex":..., "lang":...}] }}

    def load_country_schema(self, country, gt_dir):
        # Agrège les bboxes de gt/*.json -> bbox médiane par clé
        boxes_by_key = {}
        for fn in os.listdir(gt_dir):
            if not fn.lower().endswith(".json"): continue
            with open(os.path.join(gt_dir, fn), "r", encoding="utf-8") as r:
                obj = json.load(r)
            for k,v in obj.items():
                bb = v.get("bbox") or v.get("box") or v.get("bbox_xywh")
                if bb is None: 
                    continue
                boxes_by_key.setdefault(k, []).append(bb)
        fields=[]
        for k, arr in boxes_by_key.items():
            arr_np = np.array(arr, dtype=float)
            med = np.median(arr_np, axis=0).tolist()
            fields.append({"key":k, "bbox": [float(med[0]), float(med[1]), float(med[2]), float(med[3])] })
        self.templates[country] = {"template_image": None, "fields": fields}

    def set_template_image(self, country, img):
        self.templates.setdefault(country, {"template_image": None, "fields": []})
        self.templates[country]["template_image"] = img

    def fields_for_country(self, country):
        return self.templates.get(country, {}).get("fields", [])
