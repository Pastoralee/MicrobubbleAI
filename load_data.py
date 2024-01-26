import scipy.io
from os.path import join
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.nn.functional import normalize
import os
import torch
import util as ut

class Dataset(torch.utils.data.Dataset):
  def __init__(self, x_data, y_labels):
        self.y = y_labels
        self.x = x_data

  def __len__(self):
        return len(self.x)

  def __getitem__(self, index):
        X = self.x[index]
        y = self.y[index]
        return X, y

def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
    max_IQ = np.max(IQ)
    return IQ + ndimage.gaussian_filter(max_IQ * 10**(clutterdB / 20) + np.reshape(np.random.normal(size=np.prod(IQ.shape), scale=np.abs(power * impedance)), IQ.shape) * max_IQ * 10**((amplCullerdB + clutterdB) / 20), sigma=sigmaGauss)

def load_silicoFlow_data(args):
    transform = transforms.ToTensor()
    sequence = scipy.io.loadmat(join(args.pathData,"PALA_InSilicoFlow_sequence.mat"))
    Origin = sequence["PData"]["Origin"].flatten()[0][0]
    data_size = sequence["PData"]["Size"].flatten()[0][0]
    NoiseParam = {}
    NoiseParam["power"]        = -2;   # [dBW]
    NoiseParam["impedance"]    = .2;   # [ohms]
    NoiseParam["sigmaGauss"]   = 1.5;  # Gaussian filtering
    NoiseParam["clutterdB"]    = -20;  # Clutter level in dB (will be changed later)
    NoiseParam["amplCullerdB"] = 10;   # dB amplitude of clutter
    IQs, xy_pos, max_bulles = None, None, None
    for file in os.listdir(join(args.pathData,"IQ")):
        temp = scipy.io.loadmat(join(args.pathData,"IQ",file))
        if max_bulles is None:
            max_bulles = temp["ListPos"].shape[0]
        listpos = temp["ListPos"][:,[0, 2],:]
        indices_listpos = np.argsort(listpos[:,0,:], axis=0) #trier en fonction de la coordonnée y
        sorted_listpos = np.take_along_axis(listpos, indices_listpos[:, None, :], axis=0)
        xy_pos = torch.cat((xy_pos, transform(sorted_listpos)), dim=0) if xy_pos is not None else transform(sorted_listpos)
        IQs = torch.cat((IQs, transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))), dim=0) if IQs is not None else transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))
    return normalize(IQs), xy_pos, Origin, data_size, max_bulles

def load_dataset(args):
    X, Y, origin, data_size, max_bulles = load_silicoFlow_data(args)
    if args.trainType == 2:
        Y = ut.coordinates_to_mask(Y, X.shape, origin, data_size)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.testSize, shuffle=args.shuffle)
    dataset_train = Dataset(torch.unsqueeze(x_train, 1), torch.unsqueeze(y_train, 1))
    dataset_test = Dataset(torch.unsqueeze(x_test, 1), torch.unsqueeze(y_test, 1))

    kwargs = {'num_workers': args.numWorkers, 'pin_memory': True} if args.device=='cuda' else {}
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchSize, shuffle=args.shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchSize, shuffle=args.shuffle, **kwargs)
    return train_loader, test_loader, origin, data_size, max_bulles