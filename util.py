import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding par défaut, voir https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

def process_data(positions, origin, data_size, normalisation):
    result, nb_bulles = [], []
    for bubble_positions in positions: #batch_size
        pos = bubble_positions[torch.isfinite(bubble_positions)]
        pos = torch.reshape(pos, (-1, 2))
        pos[:, 0] = pos[:, 0] - origin[0]
        pos[:, 1] = pos[:, 1] - origin[2]
        pos = pos[~torch.any(pos<0, axis=1)] #enlève les valeurs inférieures à 0
        pos = pos[torch.logical_and(pos[:, 0] <= data_size[1], pos[:, 1] <= data_size[0])] #enlève les valeurs supérieures aux bordures de l'image
        if normalisation:
            pos[:, 0] /= data_size[1]
            pos[:, 1] /= data_size[0]
        result.append(pos)
        nb_bulles.append(pos.shape[0])
    del pos
    return torch.cat(result, dim=0), nb_bulles

def plot_loss(epochs, losses, pathSave):
    plt.title("Loss")
    plt.plot(epochs, losses, label="Loss par epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(pathSave + '/loss.png')
    plt.clf()

def plot_metrics(epochs, train, test, title, pathSave):
    plt.title(title)
    plt.plot(epochs, train, label="Train")
    plt.plot(epochs, test, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend(loc='best')
    plt.savefig(pathSave + '/' + title + '.png')
    plt.clf()

def coordinates_to_mask(coordinates, shape, origin, data_size):
    Y_mask = torch.zeros(shape)
    for i, coordonnees in enumerate(coordinates):
        coordonnees = coordonnees[torch.isfinite(coordonnees)]
        coordonnees = torch.reshape(coordonnees, (-1, 2))
        coordonnees[:, 0] = coordonnees[:, 0] - origin[0]
        coordonnees[:, 1] = coordonnees[:, 1] - origin[2]
        coordonnees = coordonnees[~torch.any(coordonnees<0, axis=1)]
        coordonnees = coordonnees[torch.logical_and(coordonnees[:, 0] < data_size[1]-1, coordonnees[:, 1] < data_size[0]-1)]
        for xy in coordonnees:
            xy = torch.round(xy).to(torch.int)
            Y_mask[i, xy[1], xy[0]] = 1
    return Y_mask