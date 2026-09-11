"""Fail-closed glyph extraction shared by synthetic build and title runtime."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image
SIZE=64
@dataclass(frozen=True)
class GlyphResult:
 glyphs:np.ndarray|None
 diagnostics:dict
def _fail(reason,**extra):return GlyphResult(None,{'ok':False,'reason':reason,**extra})
def _ink(image):
 a=np.asarray(image.convert('L'),dtype=np.float32)
 if min(a.shape)<4:return None,None,{'reason':'image_too_small'}
 border=np.concatenate((a[0],a[-1],a[:,0],a[:,-1]));bg=float(np.median(border));lo,hi=np.percentile(a,(2,98));dark=bool((bg-lo)>=(hi-bg));contrast=max(float(bg-lo if dark else hi-bg),16.)
 return np.clip((bg-a if dark else a-bg)/contrast,0.,1.),dark,{'background':bg,'dark_ink':dark,'contrast':contrast}
def _resize(unit):
 h,w=unit.shape;c=np.zeros((max(h,w),max(h,w)),dtype=np.float32);y=(c.shape[0]-h)//2;x=(c.shape[1]-w)//2;c[y:y+h,x:x+w]=unit
 return np.clip(np.asarray(Image.fromarray(c,mode='F').resize((SIZE,SIZE),Image.Resampling.BICUBIC),dtype=np.float32),0.,1.)
def extract_glyphs(image:Image.Image,expected_chars:int,*,min_ink_pixels:int=12)->GlyphResult:
 """Return [n,64,64] or an explicit rejection; never equal-width splits."""
 if not isinstance(expected_chars,int) or not 1<=expected_chars<=8:return _fail('invalid_expected_character_count')
 ink,dark,diag=_ink(image)
 if ink is None:return _fail(diag['reason'])
 ys,xs=np.where(ink>.18)
 if len(xs)<min_ink_pixels:return _fail('insufficient_ink',**diag,ink_pixels=int(len(xs)))
 y0,y1,x0,x1=int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1;tight=ink[y0:y1,x0:x1]
 if tight.shape[1]<expected_chars*3:return _fail('title_too_narrow_for_expected_count',**diag,bounds=[x0,y0,x1,y1])
 p=tight.sum(0);peak=float(p.max());cuts=[0];half=max(2,round(tight.shape[1]/expected_chars*.30))
 for i in range(1,expected_chars):
  center=round(i*tight.shape[1]/expected_chars);left=max(cuts[-1]+2,center-half);right=min(tight.shape[1]-2,center+half)
  if left>=right:return _fail('no_valid_separator_window',**diag)
  cut=left+int(np.argmin(p[left:right+1]))
  if p[cut]>max(.20*peak,1.5):return _fail('separator_not_low_ink',**diag,separator=int(cut),separator_mass=float(p[cut]),peak=peak)
  cuts.append(cut)
 cuts.append(tight.shape[1]);widths=np.diff(cuts)
 if widths.min()<3 or widths.max()/widths.min()>3.5:return _fail('invalid_segment_widths',**diag,widths=widths.tolist())
 glyphs=[];masses=[]
 for left,right in zip(cuts[:-1],cuts[1:]):
  u=tight[:,left:right];uy,ux=np.where(u>.18)
  if len(ux)<min_ink_pixels:return _fail('blank_or_unresolved_segment',**diag,widths=widths.tolist(),segment_ink=int(len(ux)))
  u=u[int(uy.min()):int(uy.max())+1,int(ux.min()):int(ux.max())+1];glyphs.append(_resize(u));masses.append(float(u.sum()))
 out=np.stack(glyphs).astype(np.float32)
 if out.shape!=(expected_chars,SIZE,SIZE) or not np.isfinite(out).all():return _fail('nonfinite_output')
 return GlyphResult(out,{'ok':True,**diag,'bounds':[x0,y0,x1,y1],'cuts':[int(x+x0) for x in cuts],'segment_widths':widths.tolist(),'segment_ink_mass':masses,'expected_chars':expected_chars})
