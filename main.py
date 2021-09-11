import torch.optim as optim
from os.path import join
import json
from util import DiceLoss
import model
from dataset import Loader3D, Loader2D
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from trainer import train_epoch

data_path = '/Volumes/One Touch/med_data/Task04_Hippocampus_processed'
n_epoch = 100
data_config = json.load(open(join(data_path, 'data.json')))
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

transform = None

dataset3D = Loader3D(data_path, transform)
dataset2D = Loader2D(data_path, transform)

train_validation_split = 0.9
num_train = int(len(dataset3D) * train_validation_split)
num_validation = len(dataset3D) - num_train
train3D, validation3D = torch.utils.data.random_split(dataset3D, [num_train, num_validation])

num_train = int(len(dataset2D) * train_validation_split)
num_validation = len(dataset2D) - num_train
train2D, validation2D = torch.utils.data.random_split(dataset2D, [num_train, num_validation])

trainLoader3D = DataLoader(train3D, shuffle=True, batch_size=data_config['batch_size3D'])
validLoader3D = DataLoader(validation3D, shuffle=True, batch_size=data_config['batch_size3D'])

trainLoader2D = DataLoader(train3D, shuffle=True, batch_size=data_config['batch_size2D'])
validLoader2D = DataLoader(validation3D, shuffle=True, batch_size=data_config['batch_size2D'])

model_config2D = model.get_r50_b16_config(dims=2, img_size=data_config['median_shape'],
                                          channels=data_config['channels'])
model_config3D = model.get_r50_b16_config(dims=3, img_size=data_config['median_shape'],
                                          channels=data_config['channels'])

network2D = model.VisionTransformer(model_config2D)
network3D = model.VisionTransformer(model_config3D)

optimiser3D = optim.SGD(network3D.parameters(), lr=1, momentum=0.99, nesterov=True)
optimiser2D = optim.SGD(network2D.parameters(), lr=1, momentum=0.99, nesterov=True)

dice_loss = DiceLoss(len(data_config['classes']))
ce_loss = CrossEntropyLoss()
total_loss = lambda out, lbl: 0.5 * dice_loss(out, lbl) + 0.5 * ce_loss(out, lbl)

train_loss = []
validation_loss = []
for epoch in range(n_epoch):
    loss = train_epoch(network2D, trainLoader2D, validLoader2D, device, total_loss, optimiser2D)
    train_loss.append(loss[0])
    validation_loss.append(loss[1])

