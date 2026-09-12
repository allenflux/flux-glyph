"""Export fixed calibration glyphs and PyTorch logits for independent ORT QA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from network import FontClassifier, FAMILIES
from data import dump, sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.output
    torch.set_num_threads(4)
    checkpoint=torch.load(root/'neural/model.pth',map_location='cpu',weights_only=True)
    net=FontClassifier();net.load_state_dict(checkpoint['state_dict']);net.eval()
    xs=[];source=[]
    for name in ['latin','han-freetype']:
        path=root/'data'/name/'calibration.npz';data=np.load(path,allow_pickle=False)
        indexes=np.linspace(0,len(data['x'])-1,64,dtype=int)
        xs.append(np.asarray(data['x'][indexes,None],dtype=np.float32));source.append({'path':str(path),'sha256':sha(path),'rows':indexes.tolist()})
    glyphs=np.concatenate(xs)
    with torch.inference_mode():logits=net(torch.from_numpy(glyphs)).numpy()
    np.savez_compressed(root/'parity.npz',glyphs=glyphs,logits=logits)
    dump(root/'parity-source.json',{'scope':'Fixed calibration rows, no selection or training, compare independent ORT logits.',
        'input_shape':list(glyphs.shape),'output_shape':list(logits.shape),'sources':source,'families':FAMILIES,
        'parity_sha256':sha(root/'parity.npz'),'onnx_sha256':sha(root/'neural/model.onnx')})
    print(root/'parity.npz')

if __name__=='__main__':main()
