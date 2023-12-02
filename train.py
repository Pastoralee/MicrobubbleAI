import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from prodigyopt import Prodigy
import metrics as mt

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
    pos_tensor = adjust_tensor_shapes(pred, result, device)
    return pos_tensor

def get_nbBubbles_groundTruth(positions, max_bulles):
    result = []
    for bubble_positions in positions: #batch_size
        result.append((len(bubble_positions[torch.isfinite(bubble_positions)])/2)/max_bulles)
    return torch.reshape(torch.tensor([(i) for i in result]), (-1, 1))

def train_position_model(model, args, device, train_loader, test_loader, origin, data_size, max_bulles):
    criterion = nn.MSELoss()
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_accuracy_save, train_abs_diff_total_save, test_accuracy_save, test_abs_diff_total_save, losses, epochs = [], [], [], [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for IQs, ground_truth in tqdm(train_loader):
            IQs, ground_truth = IQs.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_xy = model(torch.unsqueeze(IQs, 1))
            ground_truth_padded = process_data(ground_truth, out_xy, origin, data_size, device)
            loss_xy = criterion(out_xy, ground_truth_padded)
            loss_xy.backward()
            train_loss += loss_xy.item()
            optimizer.step()
        train_accuracy, train_abs_diff_total = mt.get_position_accuracy(model, train_loader, origin, data_size, device)
        train_accuracy_save.append(train_accuracy)
        train_abs_diff_total_save.append(train_abs_diff_total)
        test_accuracy, test_abs_diff_total = mt.get_position_accuracy(model, test_loader, origin, data_size, device)
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
    
    plt.title("Loss")
    plt.plot(epochs, losses, label="Loss par epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(args.pathSave + '/loss.png')

    plt.clf()
    plt.title("Accuracy")
    plt.plot(epochs, train_accuracy_save, label="Train")
    plt.plot(epochs, test_accuracy_save, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig(args.pathSave + '/accuracy.png')

    plt.clf()
    plt.title("RMSE")
    plt.plot(epochs, train_abs_diff_total_save, label="Train")
    plt.plot(epochs, test_abs_diff_total_save, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend(loc='best')
    plt.savefig(args.pathSave + '/rmse.png')

def train_bulle_model(model, args, device, train_loader, test_loader, max_bulles):
    criterion = nn.MSELoss()
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_erreur_bulles_save, test_erreur_bulles_save, losses, epochs = [], [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for IQs, ground_truth in tqdm(train_loader):
            IQs, ground_truth = IQs.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_numBubbles = model(torch.unsqueeze(IQs, 1))
            ground_truth_bubbles = get_nbBubbles_groundTruth(ground_truth, max_bulles).to(device=device, dtype=torch.float)
            loss_numBubbles = criterion(out_numBubbles, ground_truth_bubbles)
            loss_numBubbles.backward()
            train_loss += loss_numBubbles.item()
            optimizer.step()
        train_erreur_bulles = mt.get_bubble_accuracy(model, train_loader, device, max_bulles)
        train_erreur_bulles_save.append(train_erreur_bulles)
        test_erreur_bulles = mt.get_bubble_accuracy(model, test_loader, device, max_bulles)
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
    
    plt.title("Loss")
    plt.plot(epochs, losses, label="Loss par epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(args.pathSave + '/loss.png')

    plt.clf()
    plt.title("BubbleError")
    plt.plot(epochs, train_erreur_bulles_save, label="Train")
    plt.plot(epochs, test_erreur_bulles_save, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("BubbleError")
    plt.legend(loc='best')
    plt.savefig(args.pathSave + '/bubbleError.png')