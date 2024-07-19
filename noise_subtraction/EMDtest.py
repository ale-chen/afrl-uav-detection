import os
from PyEMD import EMD, Visualisation
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import librosa

os.chdir('C:\\Users\\Alec\\Desktop\\SIT_acoustic\\ESCAPE_FORMAT_ONECHANNEL\\testChunks')

outFile = 'C:\\Users\\Alec\\Desktop\\SIT_acoustic\\ESCAPE_FORMAT_ONECHANNEL\\testChunks\\EmdTest.wav'

filtSignal = []

files = sorted([f for f in os.listdir(os.getcwd()) if f.endswith('.wav')], key=lambda x: int(x.split('.')[0]))
print(files)
for filename in files:
    print(f"Currently processing: {filename}")
    cPath = os.path.join(os.getcwd(), filename)
    y, sr = librosa.load(cPath, sr=None) 

    emd = EMD()
    emd.emd(y)

    imfs, res = emd.get_imfs_and_residue()

    for i in range(len(imfs)):
        os.makedirs(filename[:-4], exist_ok=True)
        imf = os.path.join(filename[:-4], f'imf{i:04d}.wav')

        sf.write(imf, imfs[i], sr)
       

    filteredChunks = imfs[2:]
    filteredChunks = np.sum(filteredChunks, axis=0)

    filtSignal.append(filteredChunks)

filtSignal = np.concatenate(filtSignal)

sf.write(outFile, filtSignal, sr)