import torch
import time
import util
import json


def train(optimiser, model, data_loader, device, loss_fn):
    model.train()
    epoch_loss_train = 0.0
    for i, (img, lbl) in enumerate(data_loader):
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
    epoch_loss_validation = 0.0
    with torch.no_grad():
        for i, (img, lbl) in enumerate(data_loader):
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = loss_fn(out, lbl)
            epoch_loss_validation += loss.item()

    return epoch_loss_validation / len(data_loader)


def train_and_evaluate(epoch, device, training_dict, n_class):
    start = time.time()
    training_dict['dataset'].train()
    training_loss = train(training_dict['optimiser'], training_dict['model'], training_dict['train_loader'],
                          device, training_dict['loss_fn'])
    training_dict['train_logger'].append(training_loss)

    training_dict['dataset'].eval()
    validation_loss = evaluate(training_dict['model'], training_dict['validation_loader'], device,
                               training_dict['loss_fn'])
    training_dict['validation_logger'].append(validation_loss)
    training_dict['lr_scheduler'].step()
    elapsed_time = time.time() - start

    util.generate_example_output(training_dict['model'], training_dict['dataset'], device, epoch, n_class,
                                 dims=training_dict['dims'])

    with open(f'out/loss{training_dict["dims"]}D.json', 'w') as outfile:
        json.dump({'epochs_completed': epoch,
                   'train': training_dict['train_logger'],
                   'validation': training_dict['validation_logger']}, outfile, indent=4)

    torch.save({
        'epoch': epoch,
        'model_state_dict': training_dict['model'].state_dict(),
        'optimiser_state_dict': training_dict['optimiser'].state_dict(),
        'scheduler_state_dict': training_dict['lr_scheduler'].state_dict(),
        'train_loss': training_dict['train_logger'],
        'valid_loss': training_dict['validation_logger']
    }, training_dict['save_path'])
    print(f'{training_dict["dims"]}D - Epoch: {epoch}, Training Loss: {training_loss:.3f}, '
          f'Validation Loss: {validation_loss:.3f}, Time: {elapsed_time:.2f}')
    print(f'LR Adjusted to {training_dict["lr_scheduler"].get_lr()[0]:.3f}')