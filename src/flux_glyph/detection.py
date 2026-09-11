"""PP DB postprocessing adapted from the verified R15 adapter."""
from __future__ import annotations
import math
from typing import Any
import cv2
import numpy as np
import pyclipper
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)

def align32(value: int) -> int:
    # .NET MidpointRounding.ToEven and Python round have the same tie convention.
    return max(int(round(value / 32.0)) * 32, 32)

def resize_for_detection(rgb: np.ndarray, limit: int, limit_type: str) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = rgb.shape[:2]
    work = rgb
    padded_small = False
    if h + w < 64:
        work = np.zeros((max(32, h), max(32, w), 3), dtype=np.uint8)
        work[:h, :w] = rgb
        padded_small = True
    wh, ww = work.shape[:2]
    longest, shortest = max(wh, ww), min(wh, ww)
    if limit_type == 'max': ratio = min(1.0, limit / longest)
    elif limit_type == 'min': ratio = max(1.0, limit / shortest)
    elif limit_type == 'resize_long': ratio = limit / longest
    else: raise ValueError(f'unsupported det_limit_type {limit_type!r}')
    out_h, out_w = align32(int(wh * ratio)), align32(int(ww * ratio))
    resized = cv2.resize(work, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return resized, {'source_upright_size': [w, h], 'working_size': [ww, wh],
                     'resized_size': [out_w, out_h], 'padded_tiny_input': padded_small,
                     'limit_side_len': limit, 'limit_type': limit_type}

def normalized_nchw(rgb: np.ndarray) -> np.ndarray:
    # cv2 is used only for geometry; rgb's channel zero remains decoded R.
    f = rgb.astype(np.float32) / 255.0
    f = (f - MEAN) / STD
    return np.transpose(f, (2, 0, 1))[None, ...].astype(np.float32, copy=False)

def mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    # Same x-sort mini-box order as Paddle DBPostProcess.get_mini_boxes.
    sort = box[np.argsort(box[:, 0], kind='stable')]
    first = 0 if sort[1, 1] > sort[0, 1] else 1
    third = 2 if sort[3, 1] > sort[2, 1] else 3
    return np.array([sort[first], sort[third], sort[3 if third == 2 else 2], sort[1-first]], dtype=np.float32), min(rect[1])

def order_detection_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    tl, br = int(np.argmin(sums)), int(np.argmax(sums))
    remaining = [i for i in range(4) if i not in (tl, br)]
    if len(remaining) != 2: return points
    a, b = remaining
    tr = a if points[a,1] - points[a,0] <= points[b,1] - points[b,0] else b
    bl = b if tr == a else a
    return np.array([points[tl], points[tr], points[br], points[bl]], dtype=np.float32)

def box_score_fast(prob: np.ndarray, box: np.ndarray) -> float:
    h, w = prob.shape
    x0 = max(0, int(math.floor(float(box[:,0].min())))); x1 = min(w-1, int(math.ceil(float(box[:,0].max()))))
    y0 = max(0, int(math.floor(float(box[:,1].min())))); y1 = min(h-1, int(math.ceil(float(box[:,1].max()))))
    if x1 < x0 or y1 < y0: return 0.0
    mask = np.zeros((y1-y0+1, x1-x0+1), dtype=np.uint8)
    # Paddle/C# casts float to int here (truncate toward zero), not round.
    local = (box - np.array([x0,y0], dtype=np.float32)).astype(np.int32)
    local[:,0] = np.clip(local[:,0], 0, mask.shape[1]-1); local[:,1] = np.clip(local[:,1], 0, mask.shape[0]-1)
    cv2.fillPoly(mask, [local], 1)
    return float(cv2.mean(prob[y0:y1+1, x0:x1+1], mask=mask)[0])

def unclip(box: np.ndarray, ratio: float) -> np.ndarray | None:
    area = abs(float(cv2.contourArea(box.astype(np.float32))))
    perimeter = float(cv2.arcLength(box.astype(np.float32), True))
    if area <= 0 or perimeter <= 0: return None
    # C# reference truncates positive minibox coordinates to IntPoint.
    path = [(int(x), int(y)) for x, y in box]
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    result = offset.Execute(area * ratio / perimeter)
    if len(result) != 1 or len(result[0]) < 3: return None
    return np.asarray(result[0], dtype=np.float32)

def db_boxes(prob: np.ndarray, source_w: int, source_h: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    mask = (prob > float(cfg['det_db_thresh'])).astype(np.uint8) * 255
    if cfg['use_dilation']:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    accepted=[]; mh,mw=prob.shape
    for contour in contours[:1000]:
        box, short = mini_box(contour)
        if short < 3: continue
        score = box_score_fast(prob, box)
        if score < float(cfg['det_db_box_thresh']): continue
        expanded = unclip(box, float(cfg['det_db_unclip_ratio']))
        if expanded is None: continue
        expanded_box, expanded_short = mini_box(expanded)
        if expanded_short < 5: continue
        # DB permits source_width/source_height before the final detector filter.
        mapped = np.array([[min(source_w,max(0,round(float(x)/mw*source_w))), min(source_h,max(0,round(float(y)/mh*source_h)))] for x,y in expanded_box], dtype=np.float32)
        mapped = order_detection_points(mapped)
        # TextDetector.filter_tag_det_res then int-casts and clips to W-1/H-1.
        mapped[:,0] = np.clip(mapped[:,0].astype(np.int32), 0, source_w-1)
        mapped[:,1] = np.clip(mapped[:,1].astype(np.int32), 0, source_h-1)
        width = int(np.linalg.norm(mapped[0]-mapped[1])); height=int(np.linalg.norm(mapped[0]-mapped[3]))
        if width <= 3 or height <= 3: continue
        accepted.append({'quad_upright_xy':mapped.astype(int).tolist(),'score':score,
                         'db_minibox_map_xy':box.tolist(),'db_unclip_map_xy':expanded_box.tolist()})
    # Paddle top-to-bottom then left-to-right with the secondary line swap.
    accepted.sort(key=lambda b:(b['quad_upright_xy'][0][1], b['quad_upright_xy'][0][0]))
    for i in range(len(accepted)-1):
        for j in range(i,-1,-1):
            a,b=accepted[j],accepted[j+1]
            if abs(b['quad_upright_xy'][0][1]-a['quad_upright_xy'][0][1]) < 10 and b['quad_upright_xy'][0][0] < a['quad_upright_xy'][0][0]: accepted[j],accepted[j+1]=b,a
            else: break
    return accepted

def source_envelope(quad_list: list[list[int]], width: int, height: int) -> list[int]:
    q = np.asarray(quad_list, dtype=np.float32)
    x0,y0=np.floor(q.min(0)).astype(int); x1,y1=np.ceil(q.max(0)).astype(int)+1
    return [int(max(0,x0)),int(max(0,y0)),int(min(width,x1)),int(min(height,y1))]
