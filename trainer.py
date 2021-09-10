import torch
from os.path import join
import json

import model
from dataset import Loader3D, Loader2D
import torch.utils.data
from torch.utils.data import DataLoader

data_path = '/Volumes/One Touch/med_data/Task04_Hippocampus_processed'
data_config = json.load(open(join(data_path, 'data.json')))
device = torch.device(0)

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

model_config2D = model.get_r50_b16_config(dims=2, img_size=data_config['median_shape'], channels=data_config['channels'])
model_config3D = model.get_r50_b16_config(dims=3, img_size=data_config['median_shape'], channels=data_config['channels'])

network2D = model.VisionTransformer(model_config2D)
network3D = model.VisionTransformer(model_config3D)

