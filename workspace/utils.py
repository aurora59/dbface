import math
import numpy as np

def uexp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [uexp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([uexp(item) for item in v], v.dtype)
    
    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base
    
    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)

def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou
    
def nms(scores, bboxs, landmarks, iou=0.5):
    if scores is None or len(scores) <= 1:
        return scores, bboxs, landmarks

    scores = np.array(scores)
    bboxs  = np.array(bboxs)
    landmarks = np.array(landmarks)
    
    sc_indexs = scores.argsort()[::-1]
    
    scores = scores[sc_indexs]
    bboxs  = bboxs[sc_indexs]
    landmarks = landmarks[sc_indexs]
    
    sc_keep = []
    bbox_keep = []
    landmarks_keep = []
    
    flags = [0] * len(scores)
    for index, sc in enumerate(scores):

        if flags[index] != 0:
            continue

        sc_keep.append(scores[index])
        bbox_keep.append(bboxs[index])
        landmarks_keep.append(landmarks[index])
        
        for j in range(index + 1, len(scores)):
            if flags[j] == 0 and computeIOU(bboxs[index], bboxs[j]) > iou:
                flags[j] = 1
    return sc_keep, bbox_keep, landmarks_keep
    