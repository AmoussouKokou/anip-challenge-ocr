
import numpy as np, re

def iou(boxA, boxB):
    xA=max(boxA[0], boxB[0]); yA=max(boxA[1], boxB[1])
    xB=min(boxA[0]+boxA[2], boxB[0]+boxB[2]); yB=min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    union=boxA[2]*boxA[3]+boxB[2]*boxB[3]-inter
    return inter/union if union>0 else 0.0

def mrz_checksum(s):
    weights=[7,3,1]; total=0
    def val(c):
        if c.isdigit(): return int(c)
        if 'A'<=c<='Z': return ord(c)-55
        return 0
    for i,c in enumerate(s):
        total += val(c) * weights[i%3]
    return str(total % 10)

def build_features(ocr_items, expected_fields):
    feats={}
    ious=[]; confs=[]; miss=0; regex_ok=0; lens=[]
    for f in expected_fields:
        eb=f["bbox"]
        best=None; best_d=1e18
        ex=(eb[0]+eb[2]/2, eb[1]+eb[3]/2)
        for it in ocr_items:
            xs=[p[0] for p in it["box"]]; ys=[p[1] for p in it["box"]]
            cx,cy=sum(xs)/4,sum(ys)/4
            d=(cx-ex[0])**2+(cy-ex[1])**2
            if d<best_d: best=it; best_d=d
        if best is None: miss+=1; continue
        xs=[p[0] for p in best["box"]]; ys=[p[1] for p in best["box"]]
        bb=[min(xs),min(ys), max(xs)-min(xs), max(ys)-min(ys)]
        ious.append(iou(eb,bb))
        confs.append(best["conf"])
        txt=best["text"]
        lens.append(min(len(txt), 64))
    feats["iou_mean"]=float(np.mean(ious)) if ious else 0.0
    feats["conf_mean"]=float(np.mean(confs)) if confs else 0.0
    feats["missing_ratio"]=float(miss/max(1,len(expected_fields)))
    feats["text_len_mean"]=float(np.mean(lens)) if lens else 0.0
    return feats
