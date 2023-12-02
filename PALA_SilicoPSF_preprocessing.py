import scipy.io
from scipy import ndimage
from os.path import join
from os import getcwd
import os
import numpy as np

######################################################
############ PARAMETRES ##############################
######################################################

folder_path = join(os.path.dirname(os.path.realpath(__file__)),"PALA_data_InSilicoPSF")
print("working folder :", folder_path)

######################################################
############ ADD NOISE IN IQ #########################
######################################################

def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
    max_IQ = np.max(IQ)
    return IQ + ndimage.gaussian_filter(max_IQ * 10**(clutterdB / 20) + np.reshape(np.random.normal(size=np.prod(IQ.shape), scale=np.abs(power * impedance)), IQ.shape) * max_IQ * 10**((amplCullerdB + clutterdB) / 20), sigma=sigmaGauss)

######################################################
############ PREPROCESSING ###########################
######################################################

#Lecture fichier .mat
sequence = scipy.io.loadmat(join(folder_path,"PALA_InSilicoPSF_sequence.mat"))

P = sequence["P"]
PData = sequence["PData"]
ll = sequence["ll"]

lx = PData["Origin"][0][0][0][0] + np.arange(0,PData["Size"][0][0][0][1]-1)*PData["PDelta"][0][0][0][0]
lz = PData["Origin"][0][0][0][2] + np.arange(0,PData["Size"][0][0][0][0]-1)*PData["PDelta"][0][0][0][2]

NoiseParam = {}
NoiseParam["power"]        = -2;   # [dBW]
NoiseParam["impedance"]    = .2;   # [ohms]
NoiseParam["sigmaGauss"]   = 1.5;  # Gaussian filtering
NoiseParam["clutterdB"]    = -30;  # Clutter level in dB (will be changed later)
NoiseParam["amplCullerdB"] = 10;   # dB amplitude of clutter

IQ = scipy.io.loadmat(join(folder_path,"PALA_InSilicoPSF_IQ001.mat"))["IQ"]
IQ = PALA_AddNoiseInIQ(np.abs(IQ), **NoiseParam)

print("pre-processing done.")