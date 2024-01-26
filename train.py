import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from prodigyopt import Prodigy
import metrics as mt
import util as ut

class DynamicMSELoss(nn.Module):
    def __init__(self):
        super(DynamicMSELoss, self).__init__()
    
    def forward(self, pred, ground_truth):
        ground_truth = ground_truth.view(-1, 2)
        pred = pred.view(-1, 2)
        mask = (ground_truth.sum(dim=1) != 0)
        masked_prediction = pred[mask]
        masked_ground_truth = ground_truth[mask]
        mse_loss = nn.MSELoss()(masked_prediction, masked_ground_truth) #MSELoss
        return mse_loss

def adjust_tensor_shapes(pred, pos, device):
    max_size = max(max(torch.numel(t)/2 for t in pos), (torch.numel(pred)/len(pos))/2) #len(pos)=batch_size       
    result_list = []
    for tensor in pos:
        padding_size = int(max_size - torch.numel(tensor)/2)
        if padding_size > 0:
            padded_tensor = torch.cat((tensor, torch.zeros(padding_size, 2).to(device=device, dtype=torch.float)), dim=0)
            result_list.append(padded_tensor)
            del padded_tensor
        else:
            result_list.append(tensor)
    result_tensor = torch.stack(result_list)
    del padding_size
    del max_size
    del result_list
    return result_tensor

def process_data(positions, pred, origin, data_size, device):
    result = []
    for bubble_positions in positions: #batch_size
        pos = bubble_positions[torch.isfinite(bubble_positions)]
        pos = torch.reshape(pos, (-1, 2))
        pos[:, 0] = pos[:, 0] - origin[0]
        pos[:, 1] = pos[:, 1] - origin[2]
        pos = pos[~torch.any(pos<0, axis=1)] #enlève les valeurs inférieures à 0
        pos = pos[torch.logical_and(pos[:, 0] <= data_size[1], pos[:, 1] <= data_size[0])] #enlève les valeurs supérieures aux bordures de l'image
        pos[:, 0] /= data_size[1] #normalisation
        pos[:, 1] /= data_size[0]
        result.append(pos)
    del pos
    return adjust_tensor_shapes(pred, result, device)

def get_nbBubbles_groundTruth(positions, max_bulles):
    result = []
    for bubble_positions in positions: #batch_size
        result.append((len(bubble_positions[torch.isfinite(bubble_positions)])/2)/max_bulles)
    return torch.reshape(torch.tensor([(i) for i in result]), (-1, 1))

def train_position_map_model(model, args, train_loader, test_loader):
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_jaccard_save, train_recall_save, train_precision_save, test_jaccard_save, test_recall_save, test_precision_save, losses, epochs = [], [], [], [], [], [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for IQs, ground_truth in tqdm(train_loader):
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_xy = model(IQs)
            loss_xy = args.loss(out_xy, ground_truth)
            loss_xy.backward()
            train_loss += loss_xy.item()
            optimizer.step()
        train_jaccard, train_recall, train_precision = mt.get_position_map_accuracy(model, train_loader, args.device)
        train_jaccard_save.append(train_jaccard)
        train_recall_save.append(train_recall)
        train_precision_save.append(train_precision)
        test_jaccard, test_recall, test_precision = mt.get_position_map_accuracy(model, test_loader, args.device)
        test_jaccard_save.append(test_jaccard)
        test_recall_save.append(test_recall)
        test_precision_save.append(test_precision)
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        if scheduler is not None:
            scheduler.step()
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetMap'}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train: Jaccard: {train_jaccard}, Recall: {train_recall}, Precision: {train_precision}")
        print(f"Train: Jaccard: {test_jaccard}, Recall: {test_recall}, Precision: {test_precision}")
    ut.plot_loss(epochs, losses, args.pathSave)
    ut.plot_metrics(epochs, train_jaccard_save, test_jaccard_save, "Jaccard", args.pathSave)
    ut.plot_metrics(epochs, train_recall_save, test_recall_save, "Recall", args.pathSave)
    ut.plot_metrics(epochs, train_precision_save, test_precision_save, "Precision", args.pathSave)

def train_position_model(model, args, train_loader, test_loader, origin, data_size, max_bulles):
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_accuracy_save, train_abs_diff_total_save, test_accuracy_save, test_abs_diff_total_save, losses, epochs = [], [], [], [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for IQs, ground_truth in tqdm(train_loader):
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_xy = model(IQs)
            ground_truth_padded = process_data(ground_truth, out_xy, origin, data_size, args.device)
            loss_xy = args.loss(out_xy, ground_truth_padded)
            loss_xy.backward()
            train_loss += loss_xy.item()
            optimizer.step()
        train_accuracy, train_abs_diff_total = mt.get_position_accuracy(model, train_loader, origin, data_size, args.device)
        train_accuracy_save.append(train_accuracy)
        train_abs_diff_total_save.append(train_abs_diff_total)
        test_accuracy, test_abs_diff_total = mt.get_position_accuracy(model, test_loader, origin, data_size, args.device)
        test_accuracy_save.append(test_accuracy)
        test_abs_diff_total_save.append(test_abs_diff_total)
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        if scheduler is not None:
            scheduler.step()
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetPosition', 'max_bulles': max_bulles}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train: Accuracy: {train_accuracy}, RMSE: {train_abs_diff_total}")
        print(f"Test: Accuracy: {test_accuracy}, RMSE: {test_abs_diff_total}")
    
    ut.plot_loss(epochs, losses, args.pathSave)
    ut.plot_metrics(epochs, train_accuracy_save, test_accuracy_save, "Accuracy", args.pathSave)
    ut.plot_metrics(epochs, train_abs_diff_total_save, test_abs_diff_total_save, "RMSE", args.pathSave)

def train_bulle_model(model, args, train_loader, test_loader, max_bulles):
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_erreur_bulles_save, test_erreur_bulles_save, losses, epochs = [], [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for IQs, ground_truth in tqdm(train_loader):
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_numBubbles = model(IQs)
            ground_truth_bubbles = get_nbBubbles_groundTruth(ground_truth, max_bulles).to(device=args.device, dtype=torch.float)
            loss_numBubbles = args.loss(out_numBubbles, ground_truth_bubbles)
            loss_numBubbles.backward()
            train_loss += loss_numBubbles.item()
            optimizer.step()
        train_erreur_bulles = mt.get_bubble_accuracy(model, train_loader, args.device, max_bulles)
        train_erreur_bulles_save.append(train_erreur_bulles)
        test_erreur_bulles = mt.get_bubble_accuracy(model, test_loader, args.device, max_bulles)
        test_erreur_bulles_save.append(test_erreur_bulles)
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        if scheduler is not None:
            scheduler.step()
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetBulle', 'max_bulles': max_bulles}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train BullesErreur: {train_erreur_bulles}")
        print(f"Test BullesErreur: {test_erreur_bulles}")
    
    ut.plot_loss(epochs, losses, args.pathSave)
    ut.plot_metrics(epochs, train_erreur_bulles_save, test_erreur_bulles_save, "BubbleError", args.pathSave)