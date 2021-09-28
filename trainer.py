import torch
import time
import util
import json
import os
from os.path import join
from postprocessing import remove_all_but_the_largest_connected_component


def train(optimiser, model, data_loader, device, loss_fn):
    model.train()
    epoch_loss_train = 0.0
    for i, (img, lbl) in enumerate(data_loader):
        # Each epoch is only 250 mini batches to ensure reasonable training times
        if i == 250:
            break
        img, lbl = img.to(device), lbl.to(device)
        out = model(img)
        loss = loss_fn(out, lbl)
        epoch_loss_train += loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return epoch_loss_train / len(data_loader)


def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    loss_validation = 0.0
    dice_score = 0.0
    dice_score_component_suppression = 0.0
    with torch.no_grad():
        for i, (img, lbl) in enumerate(data_loader):
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = loss_fn(out, lbl)
            n_classes = out.shape[1]
            dice_score += util.compute_dice_score(lbl, out)
            out = torch.argmax(out, dim=1).detach().cpu().numpy()
            out = remove_all_but_the_largest_connected_component(out, n_classes)
            out = util.one_hot(out, n_classes, batch=True)
            dice_score_component_suppression += util.compute_dice_score(lbl, torch.Tensor(out).to(device))
            loss_validation += loss.item()

    do_component_suppression = dice_score_component_suppression > dice_score
    dice_score = max(dice_score, dice_score_component_suppression)
    return loss_validation / len(data_loader), dice_score / len(data_loader), do_component_suppression


def train_and_evaluate(epoch, device, training_dict, data_config):
    start = time.time()
    training_dict['dataset'].train()
    training_loss = train(training_dict['optimiser'], training_dict['model'], training_dict['train_loader'],
                          device, training_dict['loss_fn'])
    training_dict['train_logger'].append(training_loss)

    training_dict['dataset'].eval()
    validation_loss, dice_score, training_dict['do_component_suppression'] = \
        evaluate(training_dict['model'], training_dict['validation_loader'], device, training_dict['loss_fn'])
    training_dict['validation_logger'].append(validation_loss)
    training_dict['dice_logger'].append(dice_score)
    training_dict['lr_scheduler'].step()
    elapsed_time = time.time() - start

    # only save every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': training_dict['model'].state_dict(),
            'optimiser_state_dict': training_dict['optimiser'].state_dict(),
            'scheduler_state_dict': training_dict['lr_scheduler'].state_dict(),
            'train_loss': training_dict['train_logger'],
            'valid_loss': training_dict['validation_logger'],
            'dice_score': training_dict['dice_logger'],
            'do_component_suppression': training_dict['do_component_suppression']
        }, training_dict['model_path'])

    with open(join(training_dict['out_path'], f'loss{training_dict["dims"]}D.json'), 'w') as outfile:
        json.dump({'epochs_completed': epoch,
                   'train': training_dict['train_logger'],
                   'validation': training_dict['validation_logger'],
                   'dice': training_dict['dice_logger']}, outfile, indent=4)

    print(f'Epoch: {epoch}, Training Loss: {training_loss:.3f}, '
          f'Validation Loss: {validation_loss:.3f}, Dice Score: {dice_score:.3f}, Time: {elapsed_time:.2f}')
    lr = training_dict["lr_scheduler"].get_last_lr()[0]
    if lr > 0:
        print(f'{training_dict["dims"]}D - LR: {lr:.3f}, ', end='')

    # If quit file exists, gracefully exit
    if os.path.exists('terminate.txt'):
        os.remove('terminate.txt')
        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': training_dict['model'].state_dict(),
            'optimiser_state_dict': training_dict['optimiser'].state_dict(),
            'scheduler_state_dict': training_dict['lr_scheduler'].state_dict(),
            'train_loss': training_dict['train_logger'],
            'valid_loss': training_dict['validation_logger'],
            'dice_score': training_dict['dice_logger'],
            'do_component_suppression': training_dict['do_component_suppression']
        }, training_dict['model_path'])
        print("\nUser requested termination of program. \n"
              f"{training_dict['dims']}D model successfully saved after epoch {epoch}")
        quit()
