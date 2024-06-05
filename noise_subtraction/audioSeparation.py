{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "67769b61-3a59-4c3e-8239-0d4441514a4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\griff\\anaconda3\\envs\\audioSeparation\\lib\\site-packages\\tqdm\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "import noisereduce as nr\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import os\n",
    "from scipy.io import wavfile\n",
    "import librosa\n",
    "import soundfile as sf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "0df25e19-d602-4b89-ba69-425a2fdd626e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#alternative beta_loss: beta_loss='kullback-leibler' or beta_loss='itakura-saito'\n",
    "#alternative solver: solver='mu'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ff070d5b-3ebf-461f-9e15-aba7ca1bdae3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# take a signal as input, take the magnitude of the STFT of the signal, use the librosa decompose function which in turn uses the scikit-learn NMF decomposition\n",
    "# function with paramaters modified as you see fit, then takes the dot product of the components and activations to reconstruct the signal into a spectrogram\n",
    "# then take the ISTFT to convert the spectrogram back to the temporal domain and write the .wav file.\n",
    "\n",
    "# Function takes a signal and parameters as input and ouputs the reconstructed signal, a spectrogram of the signal, and the waveform of the signal \n",
    "\n",
    "def NMF_process(y, sr, filename, n_components=16, n_fft=2048, beta_loss='frobenius', solver='cd', max_iter=1000, save_dir='libNMF/'):\n",
    "    S = np.abs(librosa.stft(y, n_fft=n_fft))\n",
    "\n",
    "    comps, acts = librosa.decompose.decompose(S, sort=True, max_iter=max_iter, beta_loss=beta_loss, solver=solver, n_components=n_components)\n",
    "\n",
    "    S_approx = comps.dot(acts) \n",
    "\n",
    "    outWave = librosa.istft(S_approx) #convert back to temporal domain\n",
    "\n",
    "    wavfile.write(f\"{save_dir}{filename}_{n_components}_{solver}_{beta_loss}.wav\", sr, outWave)\n",
    "\n",
    "    plt.figure(figsize=(10, 4))\n",
    "    img = librosa.display.specshow(librosa.amplitude_to_db(S_approx,\n",
    "                                                       ref=np.max),\n",
    "                               y_axis='log', x_axis='time')\n",
    "    plt.colorbar(format='%+2.0f dB')\n",
    "    plt.title('NMF Reconstructed Spectrogram')\n",
    "    plt.savefig(f\"{save_dir}{filename}_{n_components}_{solver}_{beta_loss}.png\")\n",
    "    plt.close()\n",
    "\n",
    "    plt.figure(figsize=(10, 4))\n",
    "    wav = librosa.display.waveshow(outWave, sr=sr)\n",
    "    plt.title('NMF Reconstructed Waveform')\n",
    "    plt.savefig(f\"{save_dir}{filename}_{n_components}_{solver}_{beta_loss}_wave.png\")\n",
    "    plt.close()\n",
    "\n",
    "    return comps, acts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "8c5ae637-e5b4-4a7a-83d7-dbb822552c88",
   "metadata": {},
   "outputs": [],
   "source": [
    "#plot each component, activation, and magnitude spectrum of each component and save to a file\n",
    "def plot_comps_acts(comps, acts, sr, save_dir, filename):\n",
    "    # Plot components as waveforms\n",
    "    num_comps = comps.shape[1]\n",
    "    plt.figure(figsize=(15, 2 * num_comps))\n",
    "    for i in range(num_comps):\n",
    "        plt.subplot(num_comps, 1, i + 1)\n",
    "        plt.plot(comps[:, i])\n",
    "        plt.title(f'Component {i + 1}')\n",
    "        plt.xlabel('Frequency Bins')\n",
    "        plt.ylabel('Amplitude')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(f\"{save_dir}{filename}_components_waveforms.png\")\n",
    "    plt.close()\n",
    "\n",
    "    # Plot activations as individual waveforms\n",
    "    num_acts = acts.shape[0]\n",
    "    plt.figure(figsize=(15, 2 * num_acts))\n",
    "    for i in range(num_acts):\n",
    "        plt.subplot(num_acts, 1, i + 1)\n",
    "        plt.plot(acts[i])\n",
    "        plt.title(f'Activation {i + 1}')\n",
    "        plt.xlabel('Time Frames')\n",
    "        plt.ylabel('Activation Strength')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(f\"{save_dir}{filename}_activations_waveforms.png\")\n",
    "    plt.close()\n",
    "\n",
    "        # Example code to plot magnitude spectrum of components\n",
    "    plt.figure(figsize=(15, 2 * num_comps))\n",
    "    for i in range(num_comps):\n",
    "        plt.subplot(num_comps, 1, i + 1)\n",
    "        plt.plot(np.abs(np.fft.fft(comps[:, i])))\n",
    "        plt.title(f'Component {i + 1} Magnitude Spectrum')\n",
    "        plt.xlabel('Frequency Bins')\n",
    "        plt.ylabel('Magnitude')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(f\"{save_dir}{filename}_mag_spectrum.png\")\n",
    "    plt.close()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "72944723-4117-46ca-a200-b31148f82b15",
   "metadata": {},
   "outputs": [],
   "source": [
    "#function that combines the specified components and writes the wav file\n",
    "def combine_components(comps, acts, component_indices):\n",
    "    # Choose components and activations based on specified indices\n",
    "    source_components = comps[:, component_indices]\n",
    "    source_activations = acts[component_indices, :]\n",
    "    \n",
    "    # Reconstruct the spectrogram by combining selected components and activations\n",
    "    S_approx = source_components @ source_activations\n",
    "    \n",
    "    # Convert back to waveform\n",
    "    outWave = librosa.istft(S_approx)\n",
    "    \n",
    "    # Write output to file\n",
    "    wavfile.write(\"libNMF/output_combined.wav\", sr, outWave)\n",
    "\n",
    "    return outWave"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "83c32a13-d734-4133-bef0-e68ee747a8ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "os.chdir(f'C:\\\\Users\\griff\\AFRL_2024')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "d4617d53-06ef-4adf-bea4-cbef3cca5ac6",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename='d303sA2r03p0220210824.wav'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "8d2715d7-048f-44bf-a24c-517714de9cf7",
   "metadata": {},
   "outputs": [],
   "source": [
    "y,sr = librosa.load(filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "26f0e4cd-eba8-4913-8087-42bff0567bb5",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "comps, acts = NMF_process(y, sr, filename, n_components=8, solver='mu', beta_loss='kullback-leibler', max_iter=2000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "38cfa0b3-2b81-42d7-8106-cc20016324f3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "plot_comps_acts(comps, acts, sr, save_dir='libNMF/', filename=filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "098ed591-6eeb-45dd-8a61-41e4be632497",
   "metadata": {},
   "outputs": [],
   "source": [
    "#choose indices to combine\n",
    "idx = [1,2,3,4,5]\n",
    "out = combine_components(comps, acts, idx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "id": "8a7fd30c-32cf-41ef-8ed7-1d15f0d7689b",
   "metadata": {},
   "outputs": [],
   "source": [
    "components = [16, 32, 64, 128, 256]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "id": "6bef0bb1-bd6e-474c-93f5-e5aa22c7ccb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in components:\n",
    " NMF_process(y, sr, filename, n_components=i, solver='mu', beta_loss='itakura-saito', max_iter=2000)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
