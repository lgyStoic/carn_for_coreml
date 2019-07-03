import coremltools
import numpy as np
from PIL import Image

model =  coremltools.models.MLModel('carn.mlmodel')
inputnp = np.load('./carn/imagedate.npy')
res = model.predict({'0':inputnp})

print(res['168'].shape)
res = res['168'] * 255

np.clip(res, 0, 255, out=res)
res = res.astype('uint8')

res = np.transpose(res, (1, 2, 0))

print(res)
print(res.shape)

# res = np.ascontiguousarray(res.transpose(1,2,0))
res = Image.fromarray(res)
res.save('resultcoreml.png')