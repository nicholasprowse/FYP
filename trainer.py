import torch


def train_epoch(model, data_loader, validation_data_loader, device, loss_fn, optimiser):
    epoch_loss_train = 0.0
    for i, (img, lbl) in enumerate(data_loader):
        img, lbl = img.to(device), lbl.to(device)
        out = model(img)
        loss = loss_fn(out, lbl)
        epoch_loss_train += loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(loss.item(), end=' ')
    epoch_loss_train /= len(data_loader)

    model.eval()
    epoch_loss_validation = 0.0
    with torch.no_grad():
        for i, (img, lbl) in enumerate(validation_data_loader):
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = loss_fn(out, lbl)
            epoch_loss_validation += loss.item()
            print(loss.item())

    epoch_loss_validation /= len(validation_data_loader)

    return epoch_loss_train, epoch_loss_validation

