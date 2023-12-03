import torch
from dataclasses import dataclass
from load_data import load_dataset
from networks.UnetPosition import UnetPosition
from networks.UnetBulle import UnetBulle
from train import train_position_model
from train import train_bulle_model
from train import DynamicMSELoss

@dataclass
class Args():
    pathData: str #chemin vers les données
    pathSave: str #dossier dans lequel sauvegarder les epochs
    device: torch.device 
    testSize: float #pourcentage du dataset attribué au test [0;1]
    batchSize: int #nombre de mini batchs utilisés pendant l'entraînement
    numWorkers: int #nombre de threads pour les dataloaders
    shuffle: bool #mélanger les données
    weightDecay: float
    epochs: int
    trainType: int #0 = position, 1 = nbBulles
    loss: torch.nn.Module #DynamicMSELoss() ou torch.nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = Args(pathData = "D:\\ChefOeuvre\\data\\PALA_data_InSilicoFlow",
            pathSave = "D:\\ChefOeuvre\\save",
            device = device,
            testSize = 0.15,
            batchSize = 16,
            numWorkers = 1,
            shuffle = True,
            weightDecay = 0.01,
            epochs = 40,
            trainType = 0,
            loss = DynamicMSELoss())

#ClutterList = [-60 -40 -30 -25 -20 -15 -10]
train_loader, test_loader, origin, data_size, max_bulles = load_dataset(args)
if args.trainType == 0:
    model = UnetPosition(max_bulles)
    model = model.to(device)
    train_position_model(model, args, device, train_loader, test_loader, origin, data_size, max_bulles)
else:
    model = UnetBulle()
    model = model.to(device)
    train_bulle_model(model, args, device, train_loader, test_loader, max_bulles)