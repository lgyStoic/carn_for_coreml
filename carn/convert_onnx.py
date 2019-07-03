import importlib
import torch
import onnx
from collections import OrderedDict
module = importlib.import_module('model.carn_m')
net = module.Net(multi_scale=True, group=4)

state_dict = torch.load('../checkpoint/carn_m.pth', map_location='cpu')

# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k
#     name = k[7:] # remove "module."
#     new_state_dict[name] = v

net.load_state_dict(state_dict)
# net.load_state_dict(state_dict)
dummy_input = torch.randn(1, 3, 1000, 1000)
torch.onnx.export(net, dummy_input, "carn.onnx")

model = onnx.load('carn.onnx')
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx.save(model, 'dynamic_carn.onnx')