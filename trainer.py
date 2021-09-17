import torch


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