import importlib
import torch
from PIL import Image
import numpy as np

module = importlib.import_module('model.carn_m')
net = module.Net(multi_scale=True, group=4)
state_dict = torch.load('../checkpoint/carn_m.pth', map_location='cpu')

net.load_state_dict(state_dict)


inputnp = np.load('imagedate.npy')
inputnp = np.expand_dims(inputnp, axis=0)
inputnp = torch.from_numpy(inputnp)
output = net(inputnp)
output = output.cpu()
output = output.detach().numpy()
res = output * 255
np.clip(res, 0, 255, out=output)
outputs = output.astype('uint8')
outputs = np.squeeze(outputs, axis=0)
print(outputs.shape)
outputs = np.transpose(outputs, (1, 2, 0))

print(outputs)
print(outputs.shape)

# res = np.ascontiguousarray(res.transpose(1,2,0))
outputs = Image.fromarray(outputs)
outputs.save('resultcarnpytorchchecker.png')