import torch
import numpy as np
import train as tr
import matplotlib.pyplot as plt

def get_bubble_accuracy(model, dataloader, device, max_bulles):
    model.eval()
    erreur_bulles = 0
    for IQ, ground_truth in dataloader.dataset:
        IQ, ground_truth = IQ.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)
        out_numBubbles = model(torch.unsqueeze(torch.unsqueeze(IQ, 0), 0)) * max_bulles
        Nb_ref = int(len(ground_truth[torch.isfinite(ground_truth)])/2)
        out_numBubbles = torch.squeeze(out_numBubbles).cpu().detach().numpy() if device==torch.device("cuda") else out_numBubbles.detach().numpy()
        erreur_bulles += abs(np.round(out_numBubbles) - Nb_ref)
    erreur_bulles = erreur_bulles / len(dataloader.dataset)
    return erreur_bulles

def get_position_accuracy(model, dataloader, origin, data_size, device):
    model.eval()
    accuracy, nb_ref_total, abs_diff_total= 0, 0, 0
    for IQ, ground_truth in dataloader.dataset:
        IQ, ground_truth = IQ.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)
        out_xy = model(torch.unsqueeze(torch.unsqueeze(IQ, 0), 0))
        out_xy = torch.squeeze(out_xy)
        ground_truth = torch.unsqueeze(ground_truth, 0)
        ground_truth = tr.process_data(ground_truth, out_xy, origin, data_size, device)
        out_xy = out_xy.cpu().detach().numpy() if device==torch.device("cuda") else out_xy.detach().numpy()
        ground_truth = torch.squeeze(ground_truth).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(ground_truth).detach().numpy()
        ground_truth = ground_truth[~np.any(ground_truth<=0, axis=1)]
        ground_truth[:, 0] *= data_size[1]
        ground_truth[:, 1] *= data_size[0]
        Nb_ref = ground_truth.shape[0]
        out_xy = out_xy[:Nb_ref, :]
        out_xy[:, 0] *= data_size[1]
        out_xy[:, 1] *= data_size[0]
        abs_diff = out_xy-ground_truth
        abs_diff = np.sqrt(abs_diff[:,0]**2 + abs_diff[:,1]**2)
        abs_diff_total += np.sum(abs_diff)
        accuracy += np.sum(abs_diff < 0.35)
        nb_ref_total += Nb_ref
    accuracy = accuracy / nb_ref_total
    abs_diff_total = abs_diff_total / nb_ref_total
    return accuracy, abs_diff_total