
import cv2, numpy as np

def compute_homography(img, template):
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(img,None)
    kp2, des2 = orb.detectAndCompute(template,None)
    if des1 is None or des2 is None: return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    if len(good)<12: return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return H

def warp_to_template(img, template):
    H = compute_homography(img, template)
    if H is None: 
        return img, None
    h,w = template.shape[:2]
    warped = cv2.warpPerspective(img, H, (w,h), borderValue=(255,255,255))
    return warped, H
