import torch

from PIL import Image
inputmt = Image.open('/Users/lgy/Desktop/carntest.png').convert("RGB")

data_transform = transforms.Compose([transforms.ToTensor()])
inputmttensor = data_transform(inputmt)
inputnp = inputmttensor.numpy()