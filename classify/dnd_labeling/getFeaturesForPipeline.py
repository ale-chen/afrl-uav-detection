import os
import torch
import pandas as pd
from tqdm import tqdm
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def extract_features(chunk, sr):   
    # S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
    # chroma = librosa.feature.chroma_stft(S=np.abs(librosa.stft(S, n_fft=256)), sr=sr)
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # flat_S = S.flatten()
    # flat_chroma = chroma.flatten()
    flat_mfcc = mfcc.flatten()
    flat_mfcc_delta = mfcc_delta.flatten()
    flat_mfcc_delta2 = mfcc_delta2.flatten()
    
    # features = np.hstack((flat_S, flat_chroma, flat_mfcc, flat_mfcc_delta, flat_mfcc_delta2))
    features = np.hstack((flat_mfcc, flat_mfcc_delta, flat_mfcc_delta2))

    X_std = (features - features.min()) / (features.max() - features.min())
    X_scaled = X_std * (features.max() - features.min()) + features.min()

    return X_scaled

def normalize_and_get_feats(files):
    feats = []
    for f in tqdm(files, desc="Extracting Features", unit="file", colour="green"):
        y, sr = librosa.load(f, sr=44100)
        y = librosa.util.normalize(y)

        feats.append(extract_features(y,sr))

    return feats

def compute_spectrograms(csv_file, audio_dir, output_dir):
    df = pd.read_csv(csv_file)
    
    for filename in tqdm(df.iloc[:, 0], desc="Processing files", unit="file", colour="red"):
        # audio_path = os.path.join(audio_dir, filename)
        s_string = filename.split("/")
        spectrogram_filename = f"{s_string[-1][:-4]}_spectrogram.pt"
        output_path = os.path.join(output_dir, spectrogram_filename)

        # Load audio and comvert to spectrogram
        y,sr = librosa.load(filename, sr=44100)
        spectrogram = librosa.stft(y)
        grayscale_spectrogram = torch.FloatTensor(librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max))
       
        
        # Convert spectrogram to grayscale tensor
        grayscale_spectrogram = grayscale_spectrogram.unsqueeze(0)

        # Normalize the spectrogram
        normalized = (grayscale_spectrogram - torch.min(grayscale_spectrogram)) / (torch.max(grayscale_spectrogram) - torch.min(grayscale_spectrogram))

        # Center on 0 with range [-1, 1]
        normalized = normalized * 2 - torch.ones(normalized.shape)

        # Save normalized spectrogram
        torch.save(normalized, output_path)

    
def main():
    data_path = "/home/distasiom/Documents/Summer2024/data"
    master_csv = os.path.join(data_path, "ESCAPEII_DADS_only.csv")
    master_df = pd.read_csv(master_csv)

    files = master_df['Filename']

    filenames = []
    labels = []

    for filename in files:
        filenames.append(filename)
        # labels.append(1)
        # labels.append(get_prefix_number(filename))

    # Create a DataFrame
    df = pd.DataFrame({
        'Filename': filenames,
        # 'Label': labels
    })

    # full_paths = [os.path.join(dir, file) for file in files]

    features = normalize_and_get_feats(files)
    features_df = pd.DataFrame(features)

    res = pd.concat([master_df, features_df], axis=1)

    output_csv = '/home/distasiom/Documents/Summer2024/data/ESCAPEII_DADS_only_features.csv'

    # Write DataFrame to Excel
    res.to_csv(output_csv, index=False)


    csv_file = output_csv
    audio_dir = ""
    output_dir = '/home/distasiom/Documents/Summer2024/data/ESCAPEII_DADS_only_spectrograms'

    compute_spectrograms(csv_file, audio_dir, output_dir)

if __name__ == "__main__":
    main()