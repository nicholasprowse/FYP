import torch.optim as optim
from os.path import join
import json

import util
from util import DiceLoss
import model
from dataset import Loader3D, Loader2D
import torch.utils.data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from trainer import train, evaluate
import time
import transforms
from torchvision.transforms import Compose
import os
import numpy as np


def train_and_evaluate(epoch, device, training_dict):
    start = time.time()
    training_loss = train(training_dict['optimiser'], training_dict['model'], training_dict['train_loader'],
                          device, training_dict['loss_fn'])
    training_dict['train_logger'].append(training_loss)
    validation_loss = evaluate(training_dict['model'], training_dict['validation_loader'], device,
                               training_dict['loss_fn'])
    training_dict['validation_logger'].append(validation_loss)
    training_dict['lr_scheduler'].step()
    elapsed_time = time.time() - start

    images, labels = next(iter(training_dict['validation_loader']))
    labels = util.one_hot(labels, data_config['classes'])
    predictions = training_dict['model'](images)
    if images.ndim == 4:
        images = torch.unsqueeze(images, -1)
        labels = torch.unsqueeze(labels, -1)
        predictions = torch.unsqueeze(predictions, -1)
    images = images.numpy()
    labels = labels.numpy()
    predictions = predictions.detach().numpy()
    predictions = np.argmax(predictions, axis=1)
    predictions = util.one_hot(predictions, data_config['classes']).numpy()
    if not os.path.exists('out'):
        os.mkdir('out')
    for i in range(5):
        empty_label = np.zeros_like(labels[i])
        empty_label[0, :, :, :] = 1
        label = np.concatenate([empty_label, labels[i], predictions[i]], axis=2)
        img = np.tile(images[i, 0], (1, 3, 1))
        util.img2gif(img, 2, f"out/{training_dict['name']}_epoch{epoch}_{i}.gif", label=label)

    torch.save({
        'epoch': epoch,
        'model_state_dict': training_dict['model'].state_dict(),
        'optimiser_state_dict': training_dict['optimiser'].state_dict(),
        'scheduler_state_dict': training_dict['lr_scheduler'].state_dict(),
        'train_loss': training_dict['train_logger'],
        'valid_loss': training_dict['validation_logger']
    }, training_dict['save_path'])
    print(f'{training_dict["name"]} - Epoch: {epoch}, Training Loss: {training_loss:.3f}, '
          f'Validation Loss: {validation_loss:.3f}, Time: {elapsed_time:.2f}')
    print(f'LR Adjusted to {training_dict["lr_scheduler"].get_lr():.3f}')


def load_into_dict(training_dict):
    if os.path.isfile(training_dict['save_path']):
        check_point = torch.load(training_dict['save_path'])
        training_dict['model'].load_state_dict(check_point['model_state_dict'])
        training_dict['optimiser'].load_state_dict(check_point['optimiser_state_dict'])
        training_dict['lr_scheduler'].load_state_dict(check_point['scheduler_state_dict'])
        training_dict['train_logger'] = check_point['train_loss']
        training_dict['validation_logger'] = check_point['valid_loss']
        print(f"{training_dict['name']} checkpoint loaded, starting from epoch:", check_point['epoch'])
        return check_point['epoch']
    else:
        # Raise Error if it does not exist
        print(f"{training_dict['name']} checkpoint does not exist, starting from scratch")
        return 0


data_path = 'data/Task04_Hippocampus_processed'
n_epoch = 100
load_checkpoint = True
data_config = json.load(open(join(data_path, 'data.json')))
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

dict2D = {'name': '2D', 'save_path': join(data_path, 'model2D.pt')}
dict3D = {'name': '3D', 'save_path': join(data_path, 'model3D.pt')}

transform = Compose([
    transforms.RandomRotateAndScale(data_config),
    transforms.RandomFlip(),
    transforms.RandomBrightnessAdjustment(),
    transforms.RandomGaussianBlur(),
    transforms.RandomGaussianNoise(),
])

dataset2D = Loader2D(data_path, transform)
dataset3D = Loader3D(data_path, transform)

train_validation_split = 0.9
num_train = int(len(dataset3D) * train_validation_split)
num_validation = len(dataset3D) - num_train
train3D, validation3D = torch.utils.data.random_split(dataset3D, [num_train, num_validation],
                                                      generator=torch.Generator().manual_seed(42))

num_train = int(len(dataset2D) * train_validation_split)
num_validation = len(dataset2D) - num_train
train2D, validation2D = torch.utils.data.random_split(dataset2D, [num_train, num_validation],
                                                      generator=torch.Generator().manual_seed(42))

dict3D['train_loader'] = DataLoader(train3D, shuffle=True, batch_size=data_config['batch_size3D'])
dict3D['validation_loader'] = DataLoader(validation3D, batch_size=data_config['batch_size3D'])

dict2D['train_loader'] = DataLoader(train2D, shuffle=True, batch_size=data_config['batch_size2D'])
dict2D['validation_loader'] = DataLoader(validation2D, batch_size=data_config['batch_size2D'])

model_config2D = model.get_r50_b16_config(dims=2, img_size=data_config['shape'][1:3],
                                          channels=int(data_config['shape'][0]),
                                          num_classes=len(data_config['classes']))
model_config3D = model.get_r50_b16_config(dims=3, img_size=data_config['shape'][1:4],
                                          channels=int(data_config['shape'][0]),
                                          num_classes=len(data_config['classes']))

dict2D['model'] = model.VisionTransformer(model_config2D).to(device)
dict3D['model'] = model.VisionTransformer(model_config3D).to(device)

dict2D['optimiser'] = optim.SGD(dict2D['model'].parameters(), lr=1, momentum=0.99, nesterov=True)
dict3D['optimiser'] = optim.SGD(dict3D['model'].parameters(), lr=1, momentum=0.99, nesterov=True)

dict2D['train_logger'] = []
dict2D['validation_logger'] = []
dict3D['train_logger'] = []
dict3D['validation_logger'] = []

dict2D['lr_scheduler'] = LambdaLR(dict2D['optimiser'], lambda ep: (1 - ep / n_epoch) ** 0.9)
dict3D['lr_scheduler'] = LambdaLR(dict3D['optimiser'], lambda ep: (1 - ep / n_epoch) ** 0.9)

start_epoch = 0

if load_checkpoint:
    start_epoch = max(load_into_dict(dict2D), load_into_dict(dict3D))

dice_loss = DiceLoss(data_config['classes'])
ce_loss = CrossEntropyLoss()


def total_loss(out, target):
    dice = dice_loss(out, target, softmax=True)
    ce = ce_loss(out, target)
    return 0.5 * (ce + dice)


dict2D['loss_fn'] = dict3D['loss_fn'] = total_loss
print(f'Initial 2D loss: {evaluate(dict2D["model"], dict2D["validation_loader"], device, dict2D["loss_fn"])}')
print(f'Initial 3D loss: {evaluate(dict3D["model"], dict3D["validation_loader"], device, dict3D["loss_fn"])}')


for epoch in range(start_epoch, n_epoch):
    if epoch >= 5:
        if dict2D['validation_logger'][5] <= dict2D['validation_logger'][5]:
            train_and_evaluate(epoch, device, dict2D)
        else:
            train_and_evaluate(epoch, device, dict3D)
    else:
        train_and_evaluate(epoch, device, dict2D)
        train_and_evaluate(epoch, device, dict3D)
