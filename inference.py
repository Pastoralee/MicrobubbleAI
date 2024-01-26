from networks.UnetBulle import UnetBulle
from networks.UnetPosition import UnetPosition
from networks.UnetMap import UnetMap
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
from torchvision import transforms
import util as ut
import cv2

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

def ask_data_and_save_path():
    print("Veuillez choisir le dossier contenant vos données (.mat):")
    pathData = filedialog.askdirectory()
    print("Dossier données choisi: ", pathData)
    print("Veuillez choisir le dossier dans lequel sauvegarder vos resultats:")
    pathSave = filedialog.askdirectory() + "/"
    print("Dossier sauvegarde choisi: ", pathSave)
    return pathData, pathSave

def unet_inference():
    print("Veuillez choisir un modele pour effectuer la prédiction des coordonées:")
    filename_pos = filedialog.askopenfilename()
    print("Modele choisi: ", filename_pos)
    print("Veuillez choisir un modele pour effectuer la prédiction du nombre de bulles:")
    filename_nbbulles = filedialog.askopenfilename()
    print("Modele choisi: ", filename_nbbulles)
    pathData, pathSave = ask_data_and_save_path()

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
            coordonnees = pos_prediction[:int(nbbulles_prediction),:].tolist()
            plt.imshow(img_numpy, cmap='gray')
            plt.scatter(*zip(*coordonnees), color='red', marker='x', label='Predictions')  # Afficher les coordonnées avec des croix rouges
            plt.legend()
            plt.savefig(pathSave + f"img_{i}.png")
            plt.clf()
            result_dict[f'pred_nbBulles_img{i}'] = int(nbbulles_prediction)
            result_dict[f'pred_position_img{i}'] = coordonnees
        with open(pathSave + 'result.json', 'w') as json_file:
            json.dump(result_dict, json_file)

def map_inference():
    print("Veuillez choisir un modele pour effectuer la prédiction des coordonées:")
    filename_map = filedialog.askopenfilename()
    print("Modele choisi: ", filename_map)
    pathData, pathSave = ask_data_and_save_path()
    checkpoint_map = torch.load(filename_map, map_location=device)
    if checkpoint_map['model_name'] == 'UnetMap':
        model_map = UnetMap()
        model_map.load_state_dict(checkpoint_map['model_state_dict'])
    else:
        raise Exception("Veuillez choisir un modèle Map pour effectuer la prédiction des coordonnées")
    model_map.eval()
    model_map.to(device)
    if dataType == "Flow":
        IQs, xy_pos, origin, data_size, _ = read_flow_data(pathData)
        random_samples = torch.randint(high=IQs.shape[0], size=(nbSamples,))
        for i, num in enumerate(random_samples):
            img_tensor = IQs[num].to(device=device, dtype=torch.float)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            ground_truth = xy_pos[num].clone()
            pos_prediction = model_map(torch.unsqueeze(img_tensor, 0))
            pos_prediction = torch.squeeze(pos_prediction).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(pos_prediction).detach().numpy()
            ground_truth = ut.coordinates_to_mask(torch.unsqueeze(ground_truth, 0), img_tensor.shape, origin, data_size)
            ground_truth = torch.squeeze(ground_truth).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(ground_truth).detach().numpy()
            img_prediction = cv2.normalize(pos_prediction, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            img_ground_truth = cv2.normalize(ground_truth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(pathSave + f"predicted_img{i}.png", img_prediction)
            cv2.imwrite(pathSave + f"ground_truth_img{i}.png", img_ground_truth)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tk().withdraw()

while True:
    print("Choisissez le type d'inference:")
    print('1. Prédiction du nombre de bulles + coordonnees en parallele')
    print('2. (position map) Localisation par extraction d\'information sur les images')
    choice_train_model = input('Entrez votre choix: ')
    print('')
    if choice_train_model == '1':
        unet_inference()
        break
    elif choice_train_model == '2':
        map_inference()
        break
    print('Ce choix est invalide, veuillez choisir un nombre entre 1-2.')
