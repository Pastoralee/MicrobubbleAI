from networks.UnetBulle import UnetBulle
from networks.UnetPosition import UnetPosition
import torch
from tkinter import Tk
from tkinter import filedialog
from torchvision import transforms
import scipy.io
import os
from os.path import join
import numpy as np
from scipy import ndimage
from torch.nn.functional import normalize
import json
import matplotlib.pyplot as plt

#A CHANGER PLUS TARD
dataType = "Flow"
nbSamples = 10

def read_flow_data(pathData):
    def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
        max_IQ = np.max(IQ)
        return IQ + ndimage.gaussian_filter(max_IQ * 10**(clutterdB / 20) + np.reshape(np.random.normal(size=np.prod(IQ.shape), scale=np.abs(power * impedance)), IQ.shape) * max_IQ * 10**((amplCullerdB + clutterdB) / 20), sigma=sigmaGauss)
    transform = transforms.ToTensor()
    sequence = scipy.io.loadmat(join(pathData,"PALA_InSilicoFlow_sequence.mat"))
    Origin = sequence["PData"]["Origin"].flatten()[0][0]
    data_size = sequence["PData"]["Size"].flatten()[0][0]
    NoiseParam = {}
    NoiseParam["power"]        = -2;   # [dBW]
    NoiseParam["impedance"]    = .2;   # [ohms]
    NoiseParam["sigmaGauss"]   = 1.5;  # Gaussian filtering
    NoiseParam["clutterdB"]    = -20;  # Clutter level in dB (will be changed later)
    NoiseParam["amplCullerdB"] = 10;   # dB amplitude of clutter
    IQs, xy_pos, max_bulles = None, None, None
    for file in os.listdir(join(pathData,"IQ")):
        temp = scipy.io.loadmat(join(pathData,"IQ",file))
        if max_bulles is None:
            max_bulles = temp["ListPos"].shape[0]
        xy_pos = torch.cat((xy_pos, transform(temp["ListPos"][:,[0, 2],:])), dim=0) if xy_pos is not None else transform(temp["ListPos"][:,[0, 2],:])
        IQs = torch.cat((IQs, transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))), dim=0) if IQs is not None else transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))
    return normalize(IQs), xy_pos, Origin, data_size, max_bulles

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tk().withdraw()
print("Veuillez choisir un modele pour effectuer la prédiction des coordonées:")
filename_pos = filedialog.askopenfilename()
print("Modele choisi: ", filename_pos)
print("Veuillez choisir un modele pour effectuer la prédiction du nombre de bulles:")
filename_nbbulles = filedialog.askopenfilename()
print("Modele choisi: ", filename_nbbulles)
print("Veuillez choisir le dossier contenant vos données (.mat):")
pathData = filedialog.askdirectory()
print("Dossier données choisi: ", pathData)
print("Veuillez choisir le dossier dans lequel sauvegarder vos resultats:")
pathSave = filedialog.askdirectory() + "/"
print("Dossier sauvegarde choisi: ", pathSave)

checkpoint_pos = torch.load(filename_pos, map_location=device)
model_pos = None
if checkpoint_pos['model_name'] == 'UnetPosition':
    model_pos = UnetPosition(checkpoint_pos['max_bulles'])
    model_pos.load_state_dict(checkpoint_pos['model_state_dict'])
else:
   raise Exception("Vous n'avez pas choisi un modèle correct pour effectuer la prédiction des coordonées") 

checkpoint_nbbulles = torch.load(filename_nbbulles, map_location=device)
model_nbbulles, nbMaxBulles = None, None
if checkpoint_nbbulles['model_name'] == 'UnetBulle':
    nbMaxBulles = checkpoint_nbbulles['max_bulles']
    model_nbbulles = UnetBulle()
    model_nbbulles.load_state_dict(checkpoint_nbbulles['model_state_dict'])
else:
   raise Exception("Vous n'avez pas choisi un modèle correct pour effectuer la détection du nombre de bulles") 

model_pos.eval()
model_pos.to(device)
model_nbbulles.eval()
model_nbbulles.to(device)
if dataType == "Flow":
    IQs, xy_pos, origin, data_size, max_bulles = read_flow_data(pathData)
    random_samples = torch.randint(high=IQs.shape[0], size=(nbSamples,))
    result_dict = {}
    for i, num in enumerate(random_samples):
        img_tensor = IQs[num].to(device=device, dtype=torch.float)
        pos_prediction = model_pos(torch.unsqueeze(torch.unsqueeze(img_tensor, 0), 0))
        pos_prediction = torch.squeeze(pos_prediction).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(pos_prediction).detach().numpy()
        nbbulles_prediction = model_nbbulles(torch.unsqueeze(torch.unsqueeze(img_tensor, 0), 0)) * nbMaxBulles
        nbbulles_prediction = np.round(nbbulles_prediction.cpu().detach().numpy()) if device==torch.device("cuda") else np.round(nbbulles_prediction.detach().numpy())
        img_numpy = img_tensor.cpu().detach().numpy() if device==torch.device("cuda") else img_tensor.detach().numpy()
        pos_prediction[:, 0] *= data_size[1]
        pos_prediction[:, 1] *= data_size[0]
        #ground_truth = xy_pos[num].clone()
        #ground_truth = ground_truth[torch.isfinite(ground_truth)]
        #ground_truth = torch.reshape(ground_truth, (-1, 2))
        #ground_truth[:, 0] = ground_truth[:, 0] - origin[0]
        #ground_truth[:, 1] = ground_truth[:, 1] - origin[2]
        #ground_truth = ground_truth[~torch.any(ground_truth<0, axis=1)] #enlève les valeurs inférieures à 0
        #ground_truth = ground_truth[torch.logical_and(ground_truth[:, 0] <= data_size[1], ground_truth[:, 1] <= data_size[0])] #enlève les valeurs supérieures aux bordures de l'image
        #print(f"img n°{i}: {ground_truth}")
        plt.imshow(img_numpy, cmap='gray')
        plt.savefig(pathSave + f"img_{i}.png")
        result_dict[f'pred_nbBulles_img{i}'] = int(nbbulles_prediction)
        result_dict[f'pred_position_img{i}'] = pos_prediction[:int(nbbulles_prediction),:].tolist()
    with open(pathSave + 'result.json', 'w') as json_file:
        json.dump(result_dict, json_file)