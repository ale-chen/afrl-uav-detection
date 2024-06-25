import os
import csv
import librosa
import soundfile as sf
import pandas as pd
from tqdm.notebook import tqdm as tqdm_notebook

def calculate_hpss_ratios(directory, output_directory):
    """
    Calculate the harmonic vs percussive ratios for each .wav file in a directory.

    Args:
        directory (str): Directory path containing the .wav files.
        output_directory (str): Output directory path.

    Returns:
        None
    """
    wav_files = [file for file in tqdm_notebook(os.listdir(directory), desc="Indexing Files") if file.endswith('.wav')]
    
    results = []
    
    for file in tqdm_notebook(wav_files, desc="Processing files"):
        file_path = os.path.join(directory, file)
        y, sr = librosa.load(file_path)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        harmonic_energy = librosa.feature.rms(y=y_harmonic).mean()
        percussive_energy = librosa.feature.rms(y=y_percussive).mean()
        
        ratio = harmonic_energy / percussive_energy
        
        results.append([file, ratio])
    
    with open(os.join(output_directory,'hpss_ratios.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Harmonic/Percussive Ratio'])
        writer.writerows(results)
    
    print("HPSS ratios saved to hpss_ratios.csv")

def generate_labels_spreadsheet(wav_directory, label_spreadsheet):
    """
    Generate a new spreadsheet with labels for each .wav file in a directory.

    Args:
        wav_directory (str): Directory path containing the .wav files.
        label_spreadsheet (str): Path to the existing label spreadsheet.

    Returns:
        None
    """
    df_labels = pd.read_excel(label_spreadsheet, header=None, names=["filename", "type", "motion"])
    df_labels["identifier"] = df_labels["filename"].str.extract(r"(sA\d+r\d+)")
    
    wav_files = [file for file in os.listdir(wav_directory) if file.endswith(".wav")]
    
    df_entries = pd.DataFrame(columns=["filename", "type", "motion"])
    
    for wav_file in wav_files:
        identifier = wav_file.split("-")[0]
        
        try:
            label_row = df_labels[df_labels["identifier"] == identifier].iloc[0]
            
            entry_df = pd.DataFrame({
                "filename": [wav_file],
                "type": [label_row["type"]],
                "motion": [label_row["motion"]]
            })
            
            df_entries = pd.concat([df_entries, entry_df], ignore_index=True)
            
        except IndexError:
            print(f"No corresponding label found for file: {wav_file}")
            continue
    
    output_spreadsheet = "E:\\UAV_DISTASIO_DATA\\y\\UAV_chunk_labels.xlsx"
    df_entries.to_excel(output_spreadsheet, index=False)

def trim_or_pad_audio_files(wav_directory, max_duration_sec):
    """
    Trim or pad audio files in a directory to a specified maximum duration.

    Args:
        wav_directory (str): Directory path containing the .wav files.
        max_duration_sec (float): Maximum duration in seconds for each audio file.

    Returns:
        None
    """
    for wav in tqdm(os.listdir(wav_directory)):
        file_path = os.path.join(wav_directory, wav)
        if os.path.isfile(file_path):
            try:
                audio, sr = librosa.load(file_path)
                duration_sec = librosa.get_duration(y=audio, sr=sr)
                
                if duration_sec > max_duration_sec:
                    audio = audio[:int(max_duration_sec * sr)]
                    tqdm.write(f"Audio file trimmed: {file_path}")
                elif duration_sec < max_duration_sec:
                    pad_length = int((max_duration_sec - duration_sec) * sr)
                    audio = librosa.util.pad_center(audio, size=len(audio) + pad_length, mode='constant')
                    tqdm.write(f"Audio file padded: {file_path}")
                
                sf.write(file_path, audio, sr)
            except FileNotFoundError:
                tqdm.write(f"File not found: {file_path}")
            except librosa.util.exceptions.ParameterError as e:
                tqdm.write(f"Error loading audio file: {file_path}. Error message: {str(e)}")
        else:
            tqdm.write(f"File not found: {file_path}")