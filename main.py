import shutil

import torch.optim as optim
from os.path import join
import json
import numpy as np

import util
import model
from dataset import Loader3D, Loader2D
import torch.utils.data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from trainer import train_and_evaluate
from preprocessing import preprocess_dataset
import transforms
from torchvision.transforms import Compose
import os
import matplotlib.pyplot as plt
import matplotlib
from output import generate_test_predictions, generate_training_output
import argparse
import random
import time

parser = argparse.ArgumentParser(description="Train segmentation model.")

parser.add_argument("name", help="Name of the dataset", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
parser.add_argument("--id", help="Experiment ID to ensure each experiment has a unique output folder",
                    type=int, default=random.randint(0, 1000000))
parser.add_argument('--alpha', help="Weighting of false negatives in Tversky Loss", type=float, default=0.7)
parser.add_argument('--gamma', help="Focal factor in Tversky Loss", type=float, default=0.75)
parser.add_argument('--beta', help='Exponential factor controlling class weighting', type=float, default=0.5)
parser.add_argument('--mlp_dim', help='Number of nodes in the hidden layer of the MLP', type=int, default=3072)
parser.add_argument('--transformer_layers', help='Number of transformer layers', type=int, default=12)
parser.add_argument('--hidden_size', help='Number of channels within the transformer layers', type=int, default=768)
parser.add_argument('--num_heads', help='Number self attention heads', type=int, default=12)

args = parser.parse_args()
matplotlib.use("Agg")
raw_data_path = join('data/prepared', args.name)
data_path = join('/mnt/scratch', args.name)
out_path = f'out_new/experiment_{args.id}'
max_epoch = 1000
max_mini_batch = 10000
initial_lr = args.lr
load_checkpoint = True

if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists(out_path):
    os.mkdir(out_path)
with open(join(out_path, f'experiment.json'), 'w') as outfile:
    json.dump(args.__dict__, outfile, indent=4)
print(f"Experiment {args.id}")

start_processing = time.time()
preprocess_dataset(raw_data_path, data_path, 10 * 1024**2)
print(f'Dataset processing completed in {time.time() - start_processing} seconds')

shutil.copy(join(data_path, 'data.json'), join(out_path, 'data.json'))
data_config = json.load(open(join(data_path, 'data.json')))
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

class_frequency = np.array(data_config['class_frequency'])
class_weights = 1 / ((class_frequency ** args.beta) * (np.sum(class_frequency ** -args.beta)))
tversky_loss = util.TverskyLoss(data_config['n_classes'], weight=class_weights, alpha=args.alpha, gamma=args.gamma)
ce_loss = CrossEntropyLoss(weight=torch.tensor(class_weights, device=device).float())


def total_loss(out, target):
    dice = tversky_loss(out, target, softmax=True)
    ce = ce_loss(out, target)
    return 0.5 * (ce + dice)


transform = Compose([
    transforms.RandomRotateAndScale(data_config),
    transforms.RandomFlip(),
    transforms.RandomBrightnessAdjustment(),
    transforms.RandomGaussianBlur(),
    transforms.RandomGaussianNoise(),
])

model_config2D = model.get_r50_b16_config(dims=2, img_size=data_config['patch_size2d'],
                                          channels=int(data_config['channels']),
                                          num_classes=data_config['n_classes'],
                                          mlp_dim=args.mlp_dim,
                                          num_heads=args.num_heads,
                                          num_layers=args.transformer_layers,
                                          hidden_size=args.hidden_size)

dict2D = {'dims': 2,
          'out_path': out_path,
          'latest_model_path': join(out_path, 'latest_model2D.pt'),
          'best_model_path': join(out_path, 'best_model2D.pt'),
          'completed_epochs': 0,
          'completed_mini_batches': 0,
          'dataset': Loader2D(data_path, transform),
          'model': model.VisionTransformer(model_config2D).to(device),
          'train_logger': [],
          'validation_logger': [],
          'dice_logger': [],
          'loss_fn': total_loss}

train_validation_split = 0.9
num_train = int(len(dict2D['dataset']) * train_validation_split)
num_validation = len(dict2D['dataset']) - num_train
validation_indices = np.arange(num_validation)
train_indices = np.arange(num_validation, len(dict2D['dataset']))
train2D = torch.utils.data.Subset(dict2D['dataset'], train_indices)
validation2D = torch.utils.data.Subset(dict2D['dataset'], validation_indices)
dict2D['train_loader'] = DataLoader(train2D, shuffle=True, batch_size=data_config['batch_size2d'])
dict2D['validation_loader'] = DataLoader(validation2D, shuffle=True, batch_size=data_config['batch_size2d'])
dict2D['optimiser'] = optim.SGD(dict2D['model'].parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
dict2D['lr_scheduler'] = LambdaLR(dict2D['optimiser'], lambda ep: (1 - ep / max_mini_batch) ** 0.8)
if load_checkpoint:
    util.load_into_dict(dict2D, device)

print(f'{dict2D["dims"]}D - LR: {dict2D["lr_scheduler"].get_last_lr()[0]:.3f}, ', end='')
while dict2D['completed_epochs'] < max_epoch and dict2D['completed_mini_batches'] < max_mini_batch:
    train_and_evaluate(device, dict2D)

plt.figure()
plt.plot(dict2D['train_logger'])
plt.plot(dict2D['validation_logger'])
plt.legend(["Training Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of 2D network")
plt.ylim((0, 0.2))
plt.savefig(join(out_path, "2D_plot.pdf"))

plt.figure()
plt.plot([i[-1] for i in dict2D['dice_logger']])
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Score")
plt.savefig(join(out_path, "2D_dice_plot.pdf"))

# Remove 2D model from GPU to save memory
dict2D['model'] = dict2D['model'].cpu()
del dict2D['optimiser']
del dict2D['lr_scheduler']

model_config3D = model.get_r50_b16_config(dims=3, img_size=data_config['patch_size3d'],
                                          channels=int(data_config['channels']),
                                          num_classes=data_config['n_classes'],
                                          mlp_dim=args.mlp_dim,
                                          num_heads=args.num_heads,
                                          num_layers=args.transformer_layers,
                                          hidden_size=args.hidden_size)

dict3D = {'dims': 3,
          'out_path': out_path,
          'latest_model_path': join(out_path, 'latest_model3D.pt'),
          'best_model_path': join(out_path, 'best_model3D.pt'),
          'completed_epochs': 0,
          'completed_mini_batches': 0,
          'dataset': Loader3D(data_path, transform),
          'model': model.VisionTransformer(model_config3D).to(device),
          'train_logger': [],
          'validation_logger': [],
          'dice_logger': [],
          'loss_fn': total_loss}

num_train = int(len(dict3D['dataset']) * train_validation_split)
num_validation = len(dict3D['dataset']) - num_train
validation_indices = np.arange(num_validation)
train_indices = np.arange(num_validation, len(dict3D['dataset']))
train3D = torch.utils.data.Subset(dict3D['dataset'], train_indices)
validation3D = torch.utils.data.Subset(dict3D['dataset'], validation_indices)
dict3D['train_loader'] = DataLoader(train3D, shuffle=True, batch_size=data_config['batch_size3d'])
dict3D['validation_loader'] = DataLoader(validation3D, shuffle=True, batch_size=data_config['batch_size3d'])
dict3D['optimiser'] = optim.SGD(dict3D['model'].parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
dict3D['lr_scheduler'] = LambdaLR(dict3D['optimiser'], lambda ep: (1 - ep / max_mini_batch) ** 0.8)
if load_checkpoint:
    util.load_into_dict(dict3D, device)

if not torch.cuda.is_available():
    util.load_into_dict(dict2D, device, model='best')
    for i in range(5):
        generate_training_output(dict2D, data_config, device, i)
    quit()

print(f'{dict3D["dims"]}D - LR: {dict3D["lr_scheduler"].get_last_lr()[0]:.3f}, ', end='')
while dict3D['completed_epochs'] < max_epoch and dict3D['completed_mini_batches'] < max_mini_batch:
    train_and_evaluate(device, dict3D)

plt.figure()
plt.plot(dict3D['train_logger'])
plt.plot(dict3D['validation_logger'])
plt.legend(["Training Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of 3D network")
plt.ylim((0, 0.2))
plt.savefig(join(out_path, "3D_loss_plot.pdf"))

plt.figure()
plt.plot([i[-1] for i in dict3D['dice_logger']])
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Score")
plt.savefig(join(out_path, "3D_dice_plot.pdf"))

if dict2D['best_performance']['dice_score'][-1] > dict3D['best_performance']['dice_score'][-1]:
    final_dict = dict2D
    del dict3D
else:
    final_dict = dict3D
    del dict2D

# Reload the dictionary, as the final epoch may not have been the best
util.load_into_dict(final_dict, device, model='best')
final_dict['model'] = final_dict['model'].to(device)
generate_test_predictions(final_dict, data_config, device)

shutil.rmtree(data_path)
