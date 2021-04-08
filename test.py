from resnetV2 import ResNetV2
import torch
net = ResNetV2([4, 2, 4], 1)
img = torch.zeros((16, 3, 20, 20))

net.eval()

x, features = net(img)

print(x.shape)

print([y.shape for y in features])