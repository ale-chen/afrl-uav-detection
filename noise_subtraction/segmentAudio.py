#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.io import wavfile


# In[2]:


def segment_audio(input_file, output_dir, chunk_duration=5, sr=None):

    y, sr = librosa.load(input_file, sr=sr)
    
    #samples per chunk
    chunk_samples = chunk_duration * sr
    
    # find chunks
    num_chunks = int(np.ceil(len(y) / chunk_samples))
    
    
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, len(y))
        
        chunk = y[start_sample:end_sample]
        
        #write the chunk out
        wavfile.write(f"{output_dir}/chunk_{i + 1}.wav", sr, chunk)
        print(f"Saved {output_dir}/chunk_{i + 1}.wav")


# In[3]:


os.chdir('/Users/griffineychner/AFRL_2024')


# In[4]:


file = 'originalData/d305sA2r06p0420210826.wav'


# In[5]:


newDirName = file[13:-4]


# In[6]:


os.mkdir(newDirName)


# In[7]:


segment_audio(file, newDirName)


# In[9]:


def compute_energy(signal):
    #helper function to get energy
    return np.sum(signal ** 2)

def analyze_and_rename_chunks(input_dir):
    chunk_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')]
    energies = []

    #get energy of each chunk and combine into one array
    for chunk_file in chunk_files:
        y, sr = librosa.load(chunk_file, sr=None)
        energy = compute_energy(y)
        energies.append(energy)

    energies = np.array(energies)

    # find 25th percentile of energy
    percentile_25 = np.percentile(energies, 25)
    print(f"25th Percentile Energy: {percentile_25}")

    #sort and rename based on the energy of each file when compared with the 25th percentile
    for chunk_file, energy in zip(chunk_files, energies):
        base_name = os.path.basename(chunk_file)
        dir_name = os.path.dirname(chunk_file)
        if energy > percentile_25:
            new_name = f"{file}_drone_{base_name}"
        else:
            new_name = f"{file}_noDrone_{base_name}"
        new_path = os.path.join(dir_name, new_name)
        os.rename(chunk_file, new_path)
        print(f"Renamed {chunk_file} to {new_path}")


    return energies

input_dir = os.getcwd() + '/d303sA2r03p0220210824' 

energies = analyze_and_rename_chunks(input_dir)

