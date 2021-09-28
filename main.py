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
import transforms
from torchvision.transforms import Compose
import os
import matplotlib.pyplot as plt
import matplotlib
from output import generate_test_predictions, convert_output_to_nifti
import argparse
import random

parser = argparse.ArgumentParser(description="Train segmentation model.")

parser.add_argument("name", help="Name of the dataset", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
parser.add_argument("--id", help="Experiment ID to ensure each experiment has a unique output folder",
                    type=int, default=random.randint(0, 1000000))
parser.add_argument('--alpha', help="Weighting of false negatives in Tversky Loss", type=float, default=0.7)
parser.add_argument('--gamma', help='Exponential factor controlling class weighting', type=float, default=0.5)
parser.add_argument('--mlp_dim', help='Number of nodes in the hidden layer of the MLP', type=int, default=3072)
parser.add_argument('--transformer_layers', help='Number of transformer layers', type=int, default=12)
parser.add_argument('--hidden_size', help='Number of channels within the transformer layers', type=int, default=768)
parser.add_argument('--num_heads', help='Number self attention heads', type=int, default=12)

args = parser.parse_args()

matplotlib.use("Agg")
data_path = join('data', args.name)
out_path = f'out/experiment_{args.id}'
n_epoch = 100
decision_epoch = 100
initial_lr = args.lr
load_checkpoint = True
data_config = json.load(open(join(data_path, 'data.json')))
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
if not os.path.exists(out_path):
    os.mkdir(out_path)

with open(join(out_path, f'experiment.json'), 'w') as outfile:
    json.dump(args.__dict__, outfile, indent=4)

dict2D = {'dims': 2, 'out_path': out_path, 'model_path': join(out_path, 'model2D.pt')}
dict3D = {'dims': 3, 'out_path': out_path, 'model_path': join(out_path, 'model3D.pt')}

transform = Compose([
    transforms.RandomRotateAndScale(data_config),
    transforms.RandomFlip(),
    transforms.RandomBrightnessAdjustment(),
    transforms.RandomGaussianBlur(),
    transforms.RandomGaussianNoise(),
])

dict2D['dataset'] = Loader2D(data_path, transform)
dict3D['dataset'] = Loader3D(data_path, transform)

train_validation_split = 0.9
num_train = int(len(dict3D['dataset']) * train_validation_split)
num_validation = len(dict3D['dataset']) - num_train
train3D, validation3D = torch.utils.data.random_split(dict3D['dataset'], [num_train, num_validation],
                                                      generator=torch.Generator().manual_seed(42))

num_train = int(len(dict2D['dataset']) * train_validation_split)
num_validation = len(dict2D['dataset']) - num_train
train2D, validation2D = torch.utils.data.random_split(dict2D['dataset'], [num_train, num_validation],
                                                      generator=torch.Generator().manual_seed(42))

dict2D['train_loader'] = DataLoader(train2D, shuffle=True, batch_size=data_config['batch_size2D'])
dict2D['validation_loader'] = DataLoader(validation2D, batch_size=data_config['batch_size2D'])

dict3D['train_loader'] = DataLoader(train3D, shuffle=True, batch_size=data_config['batch_size3D'])
dict3D['validation_loader'] = DataLoader(validation3D, batch_size=data_config['batch_size3D'])

model_config2D = model.get_r50_b16_config(dims=2, img_size=data_config['shape'][0:2],
                                          channels=int(data_config['channels']),
                                          num_classes=data_config['n_classes'],
                                          mlp_dim=args.mlp_dim,
                                          num_heads=args.num_heads,
                                          num_layers=args.transformer_layers,
                                          hidden_size=args.hidden_size)

model_config3D = model.get_r50_b16_config(dims=3, img_size=data_config['shape'],
                                          channels=int(data_config['channels']),
                                          num_classes=data_config['n_classes'],
                                          mlp_dim=args.mlp_dim,
                                          num_heads=args.num_heads,
                                          num_layers=args.transformer_layers,
                                          hidden_size=args.hidden_size)


dict2D['model'] = model.VisionTransformer(model_config2D).to(device)
dict3D['model'] = model.VisionTransformer(model_config3D).to(device)

dict2D['optimiser'] = optim.SGD(dict2D['model'].parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
dict3D['optimiser'] = optim.SGD(dict3D['model'].parameters(), lr=initial_lr, momentum=0.99, nesterov=True)

dict2D['train_logger'] = []
dict2D['validation_logger'] = []
dict2D['dice_logger'] = []
dict3D['train_logger'] = []
dict3D['validation_logger'] = []
dict3D['dice_logger'] = []

dict2D['lr_scheduler'] = LambdaLR(dict2D['optimiser'], lambda ep: (1 - ep / n_epoch) ** 0.9)
dict3D['lr_scheduler'] = LambdaLR(dict3D['optimiser'], lambda ep: (1 - ep / n_epoch) ** 0.9)

dict2D['start_epoch'] = 0 if not load_checkpoint else util.load_into_dict(dict2D, device)
dict3D['start_epoch'] = 0 if not load_checkpoint else util.load_into_dict(dict3D, device)

class_frequency = np.array(data_config['class_frequency'])
class_weights = 1 / ((class_frequency ** args.gamma) * (np.sum(class_frequency ** -args.gamma)))
tversky_loss = util.TverskyLoss(data_config['n_classes'], weight=class_weights, alpha=args.alpha)
ce_loss = CrossEntropyLoss(weight=torch.tensor(data_config['class_weights'], device=device))


def total_loss(out, target):
    dice = tversky_loss(out, target, softmax=True)
    ce = ce_loss(out, target)
    return 0.5 * (ce + dice)


dict2D['loss_fn'] = total_loss
dict3D['loss_fn'] = total_loss

if not torch.cuda.is_available():
    util.generate_example_output(dict2D, data_config, device, 100)
    util.generate_example_output(dict3D, data_config, device, 100)
    # convert_output_to_nifti(out_path, data_config, 'Task04_Hippocampus', '')
    quit()

print(f'{dict2D["dims"]}D - LR: {dict2D["lr_scheduler"].get_last_lr()[0]:.3f}, ', end='')
for epoch in range(dict2D['start_epoch'], decision_epoch):
    train_and_evaluate(epoch, device, dict2D, data_config)

plt.figure()
plt.plot(dict2D['train_logger'])
plt.plot(dict2D['validation_logger'])
plt.legend(["Training Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of 2D network")
plt.savefig(join(out_path, "2D_plot.pdf"))

print(f'{dict3D["dims"]}D - LR: {dict3D["lr_scheduler"].get_last_lr()[0]:.3f}, ', end='')
for epoch in range(dict3D['start_epoch'], decision_epoch):
    train_and_evaluate(epoch, device, dict3D, data_config)

plt.figure()
plt.plot(dict3D['train_logger'])
plt.plot(dict3D['validation_logger'])
plt.legend(["Training Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of 3D network")
plt.savefig(join(out_path, "3D_plot.pdf"))

if dict2D['validation_logger'][-1] < dict3D['validation_logger'][-1]:
    final_dict = dict2D
    del dict3D
else:
    final_dict = dict3D
    del dict2D

final_dict['start_epoch'] = max(final_dict['start_epoch'], n_epoch)
for epoch in range(final_dict['start_epoch'], n_epoch):
    train_and_evaluate(epoch, device, final_dict, data_config)

generate_test_predictions(final_dict, data_config, device)