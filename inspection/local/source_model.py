"""Use the supplied source-region ONNX inference code without changing its recipe."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parent
DEFAULT_MODEL=ROOT/'models'/'strip_unet_resnet34.onnx'


def load_reference():
    path=ROOT/'source_reference.py'
    spec=importlib.util.spec_from_file_location('_source_region_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class SourceMaskOnnx:
    def __init__(self,path=DEFAULT_MODEL,width=1152,threshold=.5,min_region=.002,flip=False):
        self.path=Path(path).resolve();self.width=int(width);self.threshold=float(threshold)
        self.min_region=float(min_region);self.flip=bool(flip)
        if self.width<32 or not np.isfinite(self.threshold) or not 0<self.threshold<1:
            raise ValueError('Source model width must be >=32 and threshold must be between 0 and 1.')
        if not np.isfinite(self.min_region) or not 0<=self.min_region<1:raise ValueError('Minimum source-region area must be a fraction between 0 and 1.')
        self.reference=load_reference()
        self.reference.MODEL=self.path;self.reference.TRAIN_WIDTH=self.width
        self.reference.THRESHOLD=self.threshold;self.reference.MIN_REGION=self.min_region
        self.session=self.reference.load_session('cpu')

    def predict(self,image):
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('Source model needs a BGR photo.')
        if round(image.shape[0]*self.width/image.shape[1])<1:raise ValueError('Photo aspect ratio is too extreme for the source model width.')
        mask=self.reference.segment(self.session,image,flip=self.flip)
        if mask.shape!=image.shape[:2]:raise ValueError('Source model mask does not match original photo size.')
        return np.uint8(mask>127)*255

    def settings(self):
        return dict(model=str(self.path),train_width=self.width,threshold=self.threshold,
                    minimum_region_fraction=self.min_region,mirrored_pass=self.flip,
                    preprocessing='aspect-preserving resize; reflected padding to multiple of 32; RGB ImageNet normalization',
                    postprocessing='sigmoid; optional mirrored probability average; full-size resize; area filter; hole fill',
                    inference_script=str(ROOT/'source_reference.py'),provider='CPUExecutionProvider')
