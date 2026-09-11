"""Frozen R9 foreground normalization, copied without formula changes."""
from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Any

def normalize_foreground(image:Image.Image)->tuple[Image.Image|None,dict[str,Any]]:
 """Fixed V2 preprocessing, symmetric for bank/query, with no fallback font."""
 a=np.asarray(image.convert('L'),dtype=np.float32)
 if a.ndim!=2 or min(a.shape)<4:return None,{'ok':False,'reason':'image_too_small'}
 border=np.concatenate((a[0],a[-1],a[:,0],a[:,-1])); bg=float(np.median(border))
 dark=float(bg-a.min()); light=float(a.max()-bg); dark_ink=dark>=light
 departure=np.maximum(bg-a,0) if dark_ink else np.maximum(a-bg,0)
 contrast=float(departure.max()); threshold=.18*contrast
 mask=departure>=threshold
 ys,xs=np.where(mask); before=float(mask.mean())
 if contrast<16:return None,{'ok':False,'reason':'low_max_contrast','background':bg,'contrast':contrast,'foreground_occupancy_before':before}
 if len(xs)<12:return None,{'ok':False,'reason':'insufficient_foreground','background':bg,'contrast':contrast,'foreground_occupancy_before':before,'foreground_pixels':int(len(xs))}
 y0,y1,x0,x1=int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
 tight=a[y0:y1,x0:x1]; narrow_scale=1.0
 if tight.shape[1]<4:
  narrow_scale=4.0/tight.shape[1]
  tight=np.asarray(Image.fromarray(np.clip(np.rint(tight),0,255).astype(np.uint8),'L').resize((4,max(1,round(tight.shape[0]*narrow_scale))),Image.Resampling.BICUBIC),dtype=np.float32)
 padded=np.full((tight.shape[0]+8,tight.shape[1]+8),bg,dtype=np.float32);padded[4:-4,4:-4]=tight
 padded_u8=np.clip(np.rint(padded),0,255).astype(np.uint8)
 return Image.fromarray(padded_u8,'L').convert('RGB'),{'ok':True,'background':bg,'dark_ink':dark_ink,'contrast':contrast,'threshold':threshold,'foreground_occupancy_before':before,'foreground_occupancy_after':float(((np.maximum(bg-padded,0) if dark_ink else np.maximum(padded-bg,0))>=threshold).mean()),'bbox':[x0,y0,x1,y1],'narrow_width_upscale':narrow_scale,'padded_size':[int(padded.shape[1]),int(padded.shape[0])]}
