import torch
import time
import util
import json
import os
from os.path import join
from postprocessing import remove_all_but_the_largest_connected_component


epoch_length = 250


def train(training_dict, device):
    optimiser = training_dict['optimiser']
    model = training_dict['model']
    data_loader = training_dict['train_loader']
    loss_fn = training_dict['loss_fn']
    model.train()
    epoch_loss_train = 0.0
    mini_batches = 0
    for i, (img, lbl) in enumerate(data_loader):
        # Each epoch is only 250 mini batches to ensure reasonable training times
        if i == epoch_length:
            break
        img, lbl = img.to(device), lbl.to(device)
        out = model(img)
        loss = loss_fn(out, lbl)
        epoch_loss_train += loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        mini_batches += 1
        training_dict['completed_mini_batches'] += 1
        training_dict['lr_scheduler'].step()
        lr = training_dict['lr_scheduler'].get_last_lr()[0]
        if lr == 0 or type(lr) == complex:
            break

    return epoch_loss_train / mini_batches


def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    loss_validation = 0.0
    dice_score = None
    dice_score_component_suppression = None
    mini_batches = 0
    with torch.no_grad():
        for i, (img, lbl) in enumerate(data_loader):
            if i == epoch_length:
                break
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = loss_fn(out, lbl)
            n_classes = out.shape[1]
            if dice_score is None:
                dice_score = util.compute_dice_score(lbl.detach(), out.detach())
            else:
                dice_score += util.compute_dice_score(lbl.detach(), out.detach())
            out = torch.argmax(out, dim=1).detach().cpu().numpy()
            out = remove_all_but_the_largest_connected_component(out, n_classes)
            out = util.one_hot(out, n_classes, batch=True)
            if dice_score_component_suppression is None:
                dice_score_component_suppression = util.compute_dice_score(lbl.detach(), torch.Tensor(out).to(device))
            else:
                dice_score_component_suppression += util.compute_dice_score(lbl.detach(), torch.Tensor(out).to(device))
            loss_validation += loss.item()
            mini_batches += 1

    do_component_suppression = dice_score_component_suppression[-1] > dice_score[-1]
    if do_component_suppression:
        dice_score = dice_score_component_suppression
    return loss_validation / mini_batches, dice_score / mini_batches, do_component_suppression


def train_and_evaluate(device, training_dict):
    start = time.time()
    training_dict['dataset'].train()
    training_loss = train(training_dict, device)
    training_dict['train_logger'].append(training_loss)

    training_dict['dataset'].eval()
    validation_loss, dice_score, training_dict['do_component_suppression'] = \
        evaluate(training_dict['model'], training_dict['validation_loader'], device, training_dict['loss_fn'])
    training_dict['validation_logger'].append(validation_loss)
    training_dict['dice_logger'].append(dice_score.tolist())
    training_dict['completed_epochs'] += 1

    if 'best_performance' not in training_dict or training_dict['best_performance']['dice_score'][-1] > dice_score[-1]:
        training_dict['best_performance'] = {'completed_epochs': training_dict['completed_epochs'],
                                             'train_loss': training_loss, 'validation_loss': validation_loss,
                                             'dice_score': dice_score.tolist(),
                                             'component_suppression': training_dict['do_component_suppression']}
        save_model(training_dict)

    with open(join(training_dict['out_path'], f'loss{training_dict["dims"]}D.json'), 'w') as outfile:
        json.dump({'epochs_completed': training_dict["completed_epochs"],
                   'mini_batches_completed': training_dict["completed_mini_batches"],
                   'best_performance': training_dict['best_performance'],
                   'train': training_dict['train_logger'],
                   'validation': training_dict['validation_logger'],
                   'dice': training_dict['dice_logger']}, outfile, indent=4)

    elapsed_time = time.time() - start
    print(f'Epoch: {training_dict["completed_epochs"]}, Training Loss: {training_loss:.3f}, '
          f'Validation Loss: {validation_loss:.3f}, Dice Score: {dice_score[-1]:.3f}, Time: {elapsed_time:.2f}')
    lr = training_dict["lr_scheduler"].get_last_lr()[0]
    if lr > 0:
        print(f'{training_dict["dims"]}D - LR: {lr:.3f}, ', end='')

    # If quit file exists, gracefully exit
    if os.path.exists('terminate.txt'):
        os.remove('terminate.txt')
        save_model(training_dict)
        print("\nUser requested termination of program. \n"
              f"{training_dict['dims']}D model successfully saved after {training_dict['completed_epochs']} epochs")
        quit()


def save_model(training_dict):
    torch.save({
        'completed_epochs': training_dict['completed_epochs'],
        'completed_mini_batches': training_dict['completed_mini_batches'],
        'model_state_dict': training_dict['model'].state_dict(),
        'optimiser_state_dict': training_dict['optimiser'].state_dict(),
        'scheduler_state_dict': training_dict['lr_scheduler'].state_dict(),
        'train_loss': training_dict['train_logger'],
        'valid_loss': training_dict['validation_logger'],
        'dice_score': training_dict['dice_logger'],
        'do_component_suppression': training_dict['do_component_suppression'],
        'best_performance': training_dict['best_performance']
    }, training_dict['model_path'])
