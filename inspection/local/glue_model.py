"""Reference two-class SegFormer ONNX preprocessing and prediction."""
from pathlib import Path
import cv2
import numpy as np

def segformer_input(image,width=1024,height=64):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)

def segformer_mask(logits,shape):
    if logits.ndim!=4 or logits.shape[:2]!=(1,2):
        raise ValueError(f'Expected SegFormer output [1,2,H,W], got {logits.shape}.')
    if not np.isfinite(logits).all():raise ValueError('Model returned nonfinite values.')
    labels=np.argmax(logits,axis=1)[0].astype(np.uint8)*255
    return cv2.resize(labels,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)

class SegformerOnnx:
    """Match the user's two-class SegFormer training/inference recipe."""
    def __init__(self,path,width=1024,height=64):
        self.path=str(Path(path).resolve());self.width=int(width);self.height=int(height)
        if min(self.width,self.height)<2:raise ValueError('Invalid model input dimensions.')
        try:import onnxruntime as ort
        except ImportError as exc:raise ValueError('ONNX Runtime is missing. Install requirements.txt into the application environment.') from exc
        self.session=ort.InferenceSession(self.path,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs();outputs=self.session.get_outputs()
        if len(inputs)!=1 or not outputs:raise ValueError('Expected one SegFormer input and at least one output.')
        self.input_name=inputs[0].name;self.output_name=outputs[0].name
        if inputs[0].type!='tensor(float)':raise ValueError('Model must accept float32 input.')
        shape=inputs[0].shape
        expected=(1,3,self.height,self.width)
        if len(shape)!=4 or any(isinstance(a,int) and a!=b for a,b in zip(shape,expected)):
            raise ValueError(f'Model input {shape} does not match configured NCHW {expected}.')

    def predict(self,image):
        logits=self.session.run([self.output_name],{self.input_name:segformer_input(image,self.width,self.height)})[0]
        return segformer_mask(logits,image.shape)

    def settings(self):
        return dict(path=self.path,width=self.width,height=self.height,
                    mean=[.485,.456,.406],std=[.229,.224,.225],input_name=self.input_name,output_name=self.output_name,
                    input='float32 NCHW normalized RGB',output='argmax over 2 classes; foreground class 1; nearest resize')
