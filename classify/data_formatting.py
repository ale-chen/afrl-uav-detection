import os
import csv
import librosa
import soundfile as sf
import pandas as pd
from tqdm.notebook import tqdm as tqdm
import numpy as np

def get_drone_type(filename):
    drone_types = {
        'Inspired': 0,
        'Matrice-': 1,
        '-Phantom': 2,
        'Matrice+Phantom': 3
    }
    for key in drone_types:
        if key in filename:
            return drone_types[key]
    return None

def split_wav_file(input_file, output_dir, chunk_duration=0.25):
    # Crashes out randomly *sometimes*
    y, sr = librosa.load(input_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(duration // chunk_duration)
    
    output_filenames = []
    output_labels = []
    
    for i in range(num_chunks):
        start = int(i * chunk_duration * sr)
        end = int((i + 1) * chunk_duration * sr)
        chunk = y[start:end]
        
        if len(chunk) < chunk_duration * sr:
            padding = np.zeros(int(chunk_duration * sr) - len(chunk))
            chunk = np.concatenate((chunk, padding))
        
        input_filename = os.path.basename(input_file)
        output_filename = f"{os.path.splitext(input_filename)[0]}_{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        sf.write(output_path, chunk, sr)
        
        output_filenames.append(output_filename)
        output_labels.append(get_drone_type(input_filename))
    
    return output_filenames, output_labels

def process_files(input_excel, source_dir, output_dir, output_excel):
    df = pd.read_excel(input_excel)
    
    all_output_filenames = []
    all_output_labels = []
    
    for filename in tqdm(df.iloc[:, 0], desc="Processing files", unit="file"):
        input_file = os.path.join(source_dir, filename)
        output_filenames, output_labels = split_wav_file(input_file, output_dir)
        all_output_filenames.extend(output_filenames)
        all_output_labels.extend(output_labels)
    
    output_df = pd.DataFrame({'Filename': all_output_filenames, 'Label': all_output_labels})
    output_df.to_excel(output_excel, index=False)

def calculate_hpss_ratios(directory, output_directory):
    """
    Calculate the harmonic vs percussive ratios for each .wav file in a directory.

    Args:
        directory (str): Directory path containing the .wav files.
        output_directory (str): Output directory path.

    Returns:
        None
    """
    wav_files = [file for file in tqdm(os.listdir(directory), desc="Indexing Files") if file.endswith('.wav')]
    
    results = []
    
    for file in tqdm(wav_files, desc="Processing files"):
        file_path = os.path.join(directory, file)
        y, sr = librosa.load(file_path)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        harmonic_energy = librosa.feature.rms(y=y_harmonic).mean()
        percussive_energy = librosa.feature.rms(y=y_percussive).mean()
        
        ratio = harmonic_energy / percussive_energy
        
        results.append([file, ratio])
    
    with open(os.path.join(output_directory,'hpss_ratios.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Harmonic/Percussive Ratio'])
        writer.writerows(results)
    
    print("HPSS ratios saved to hpss_ratios.csv")

def select_data_by_hpss_ratio(directory):
    """
    Generate a new csv based on the harmonic vs percussive ratio of each .wav file in a directory.

    Args:
        directory (str): Directory path containing the csv file.

    Returns:
        None
    """
    wav_files = []
    cutoff = 1.25

    with open(os.path.join(directory,'hpss_ratios.csv'), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        wav_files += [[row[0],' ',' '] for row in csv_reader if float(row[1]) >= cutoff]

    df_labels = pd.read_excel("E:\\UAV_DISTASIO_DATA\\y\\UAV_chunk_labels.xlsx", header=None, names=["filename", "type", "motion"])
    df_labels["identifier"] = df_labels["filename"].str.extract(r"(sA\d+r\d+)")
    
    df_entries = pd.DataFrame(columns=["filename", "type", "motion"])
    
    for wav_file in wav_files:
        identifier = wav_file[0].split("-")[0]
        
        try:
            label_row = df_labels[df_labels["identifier"] == identifier].iloc[0]
            
            entry_df = pd.DataFrame({
                "filename": [wav_file[0]],
                "type": [label_row["type"]],
                "motion": [label_row["motion"]]
            })
            
            df_entries = pd.concat([df_entries, entry_df], ignore_index=True)
            
        except IndexError:
            print(f"No corresponding label found for file: {wav_file[0]}")
            continue
    
    output_spreadsheet = "E:\\UAV_DISTASIO_DATA\\y\\UAV_chunk_labels_reduced.xlsx"
    df_entries.to_excel(output_spreadsheet, index=False)
    
    print(f"New label file of sound chunks with hpss ratio greater than {cutoff}")

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









if __name__ == '__main__':
    # Example usage
    input_excel = 'input_filenames.xlsx'
    source_dir = 'path/to/source/directory'
    output_dir = 'path/to/output/directory'
    output_excel = 'output_mapping.xlsx'