import numpy as np
import scipy.io
import os
import sys
from scipy import signal


##################  CORRECT USAGE ################## 
if len(sys.argv) !=3:
	print("Usage : python InVivo_preprocess.py InVivo_folder_path save_folder_path")
	exit(1)
############################################################################################################

################## SVD FILTER ##################
def SVDfilter(IQ,cutoff):
	initsize = IQ.shape
	initsize_x = initsize[0]
	initsize_y = initsize[1]
	initsize_z = initsize[2]

	if cutoff[-1] > initsize[-1]:
		cutoff = [cutoff[0],initsize[-1]]

	if len(cutoff) == 1:
		cutoff = [cutoff[0],initsize[-1]]

	elif len(cutoff) == 2:
		cutoff = [cutoff[0],cutoff[1]] 

	if cutoff == [1,IQ.shape[2]] or cutoff[0] < 2:
		IQf = IQ
		return IQf

	cutoff[0] = cutoff[0] - 1
	cutoff[1] = cutoff[1] - 1 # in MatLab, array[200] give you access to the 200th element, unlike Python


	X = np.reshape(IQ,(initsize_x*initsize_y,initsize_z))# % Reshape into Casorati matric
	

	U,S,Vh = scipy.linalg.svd(np.dot(X.T,X)) #calculate svd of the autocorrelated Matrix
	V = np.dot(X,U) # Calculate the singular vectors.

	Reconst = np.dot(V[:,cutoff],U[:,cutoff].T) # Singular value decomposition

	IQf = np.reshape(Reconst,(initsize_x,initsize_y,initsize_z)) #% Reconstruction of the final filtered matrix
	return np.absolute(IQf)
############################################################################################################

### BEGINNING PRE PROCESSING ###

IQ_path = os.path.join(sys.argv[1],"IQ")

save_path = sys.argv[2]

if not(os.path.exists(save_path)):
	os.mkdir(save_path)

for file in os.listdir(IQ_path):
	file_path = os.path.join(IQ_path,file)
	mat_file = scipy.io.loadmat(file_path)
	# mat_file is a dict
	IQ = mat_file["IQ"]
	framerate = mat_file["UF"]["FrameRateUF"][0][0][0][0]
	#print(framerate)
	cutoff = [50,framerate]

	# SVD filter
	bulles = SVDfilter(IQ,cutoff)
	#print(bulles)
	# Temporal filter
	but_b,but_a = scipy.signal.butter(2,[50/(framerate*0.5),249/(framerate*0.5)],btype='bandpass')
	#print(but_b.shape,but_a.shape,bulles.shape)
	bulles = signal.lfilter(but_b,but_a,bulles,axis=2)
	file = file.replace(".mat","")
	save_name = "python" + file
	np.save(os.path.join(save_path,save_name),bulles)

print("Job's done")