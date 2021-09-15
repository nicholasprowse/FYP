import torch


def train_epoch(model, data_loader, validation_data_loader, device, loss_fn, optimiser):
    print('Training')
    model.train()
    epoch_loss_train = 0.0
    for i, (img, lbl) in enumerate(data_loader):
        img, lbl = img.to(device), lbl.to(device)
        out = model(img)
        loss = loss_fn(out, lbl)
        epoch_loss_train += loss.item()
        print(f', Total: {loss.item():.3f}')
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    epoch_loss_train /= len(data_loader)

    print('Evaluating')
    model.eval()
    epoch_loss_validation = 0.0
    with torch.no_grad():
        for i, (img, lbl) in enumerate(validation_data_loader):
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = loss_fn(out, lbl)
            epoch_loss_validation += loss.item()
            print(f', Total: {loss.item():.3f}')

    epoch_loss_validation /= len(validation_data_loader)

    return epoch_loss_train, epoch_loss_validation
