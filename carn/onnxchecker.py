import onnx
import caffe2.python.onnx.backend as backend
from PIL import Image
import torch
# import torchvision.transforms as transforms

import numpy as np
model = onnx.load("carn.onnx")
onnx.checker.check_model(model)
rep = backend.prepare(model)

inputnp = np.load('imagedate.npy')
inputnp = np.expand_dims(inputnp, axis=0)
outputs = rep.run(inputnp)
print(outputs[0])
res = outputs[0] * 255
np.clip(res, 0, 255, out=res)
res = res.astype('uint8')
res = np.squeeze(res, axis=0)
print(res.shape)
res = np.transpose(res, (1, 2, 0))

print(res)
print(res.shape)

# res = np.ascontiguousarray(res.transpose(1,2,0))
res = Image.fromarray(res)
res.save('resultcarnonnxchecker.png')