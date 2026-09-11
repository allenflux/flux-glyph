"""PP-OCR CPU runtime. Recognition timesteps are coarse anchors, never glyph boxes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from .detection import resize_for_detection, normalized_nchw, db_boxes, source_envelope

cv2.setNumThreads(1)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / 'models' / 'pp'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(directory, kind, contract):
    spec = contract['models'][kind]
    path = directory / 'onnx' / f'paddle_ocr_{kind}.onnx'
    if path.stat().st_size != spec['size_bytes'] or sha(path) != spec['sha256']:
        raise ValueError(f'PP {kind} model checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    engine = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    inputs, outputs = engine.get_inputs(), engine.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != spec['io']['inputs'][0]['name'] or outputs[0].name != spec['io']['outputs'][0]['name']:
        raise ValueError('PP ONNX input/output name or count mismatch')
    rank = {'cls': 2, 'rec': 3, 'det': 4}[kind]
    if inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4 or len(outputs[0].shape) != rank or inputs[0].shape[1] != 3:
        raise ValueError('PP ONNX type/rank mismatch')
    return engine


def contract_at(directory):
    directory = Path(directory or DEFAULT_MODEL_DIR)
    contract = json.loads((directory / 'paddle_ocr_delivery.contract.json').read_text())
    if contract['adapter_contract']['input_color_order'] != 'RGB_passthrough_to_paddle_v2':
        raise ValueError('Unsupported PP color contract')
    return directory, contract


def prepare(image, height, width):
    a = np.asarray(image if image.mode == 'RGB' else image.convert('RGB'))
    content_width = min(width, max(1, int(np.ceil(a.shape[1] * height / a.shape[0]))))
    a = cv2.resize(a, (content_width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.
    a = (a - .5) / .5
    x = np.zeros((3, height, width), dtype=np.float32)
    x[:, :, :content_width] = a.transpose(2, 0, 1)
    return x, content_width


def decode_ctc(scores, characters):
    if scores.ndim != 2 or scores.shape[1] != len(characters) or not np.isfinite(scores).all():
        raise ValueError('Invalid CTC scores')
    ids, probabilities = scores.argmax(1), scores.max(1)
    tokens, first_scores = [], []
    last = 0
    for step, (label, probability) in enumerate(zip(ids, probabilities)):
        label, probability = int(label), float(probability)
        if label == 0:
            last = 0
            continue
        if label != last:
            tokens.append({'character': characters[label], 'index': len(tokens), 'start_step': step,
                           'end_step': step + 1, 'peak_step': step, 'confidence': probability})
            first_scores.append(probability)
        else:
            token = tokens[-1]
            token['end_step'] = step + 1
            if probability > token['confidence']:
                token['confidence'], token['peak_step'] = probability, step
        last = label
    return ''.join(x['character'] for x in tokens), float(np.mean(first_scores)) if first_scores else 0., tokens


class PPRegionDetector:
    def __init__(self, model_dir=None, verify=True):
        directory, contract = contract_at(model_dir)
        self.session = session(directory, 'det', contract)
        keys = ('det_limit_side_len','det_limit_type','det_db_thresh','det_db_box_thresh','det_db_unclip_ratio','use_dilation','det_db_score_mode')
        self.config = {k: contract['effective_paddleocr_args'][k] for k in keys}
        expected = dict(det_limit_side_len=960,det_limit_type='max',det_db_thresh=.3,det_db_box_thresh=.6,det_db_unclip_ratio=1.5,use_dilation=False,det_db_score_mode='fast')
        if self.config != expected:
            raise ValueError('Unsupported DB configuration')

    def detect(self, image):
        rgb = np.asarray(image.convert('RGB'))
        resized, _ = resize_for_detection(rgb, 960, 'max')
        output = self.session.run(None, {'x': normalized_nchw(resized)})[0]
        if output.ndim != 4 or output.shape[:2] != (1,1) or not np.isfinite(output).all():
            raise ValueError('Invalid DB output')
        found = db_boxes(output[0,0], image.width, image.height, self.config)
        return [{'quad': b['quad_upright_xy'], 'source_bbox': source_envelope(b['quad_upright_xy'], image.width, image.height), 'score': b['score']} for b in found]


class PPReader:
    def __init__(self, model_dir=None, verify=True):
        directory, contract = contract_at(model_dir)
        dictionary = directory / 'charset/ppocr_keys_v1.txt'
        if sha(dictionary) != contract['dictionary']['sha256'] or dictionary.stat().st_size != contract['dictionary']['size_bytes']:
            raise ValueError('PP dictionary checksum mismatch')
        self.characters = [''] + dictionary.read_text(encoding='utf-8').splitlines() + [' ']
        if len(self.characters) != 6625:
            raise ValueError('Unexpected dictionary size')
        self.cls = session(directory, 'cls', contract)
        self.rec = session(directory, 'rec', contract)

    def read(self, images):
        if not isinstance(images, list):
            raise TypeError('Expected image list')
        if not images:
            return []
        images = [x if x.mode == 'RGB' else x.convert('RGB') for x in images]
        order = sorted(range(len(images)), key=lambda i: images[i].width / images[i].height)
        results = [None] * len(images)
        for start in range(0, len(order), 6):
            indices = order[start:start+6]
            batch = [images[i] for i in indices]
            logits = self.cls.run(None, {'x': np.stack([prepare(x,48,192)[0] for x in batch])})[0]
            if logits.shape != (len(batch),2) or not np.isfinite(logits).all():
                raise ValueError('Invalid CLS output')
            rotations = [180 if int(p.argmax()) == 1 and float(p.max()) > .9 else 0 for p in logits]
            batch = [x.transpose(Image.Transpose.ROTATE_180) if rotations[i] else x for i,x in enumerate(batch)]
            width = int(48 * max(320/48, max(x.width/x.height for x in batch)))
            # Bound pathological long text lines on a 2 GB host. Preserve their
            # detection boxes and reject font inference; never silently truncate.
            if width > 4096:
                for i in indices:
                    # Isolate long lines, so neighbouring short lines still run.
                    x = images[i]
                    if 48*x.width/x.height > 4096:
                        results[i] = {'text':'','confidence':0.,'tokens':[], 'metadata':{'timesteps':0,'input_width':0,'content_width':0,'original_width':x.width,'original_height':x.height,'orientation_degrees':0,'reason':'text_line_too_wide'}}
                    else:
                        results[i] = self.read([x])[0]
                continue
            arrays, contents = zip(*(prepare(x,48,width) for x in batch))
            output = self.rec.run(None, {'x': np.stack(arrays)})[0]
            if output.ndim != 3 or output.shape[0] != len(batch) or output.shape[2] != 6625 or not np.isfinite(output).all():
                raise ValueError('Invalid REC output')
            for j,i in enumerate(indices):
                text, confidence, tokens = decode_ctc(output[j], self.characters)
                x = batch[j]
                results[i] = {'text':text,'confidence':confidence,'tokens':tokens,
                    'metadata':{'timesteps':int(output.shape[1]),'input_width':width,'content_width':contents[j],
                                'original_width':x.width,'original_height':x.height,'orientation_degrees':rotations[j],
                                'ctc_positions_are_coarse_anchors':True}}
        return results
