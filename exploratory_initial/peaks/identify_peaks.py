import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import cwt, find_peaks
from scipy.stats import norm
import os
import pandas as pd

def wavelet_peak_detection(signal, fs):
    # Apply continuous wavelet transform (CWT)
    scales = np.arange(1, 101)
    coefs, freqs = cwt(signal, scales, 'morl', 1/fs)
    
    # Identify peaks in the wavelet coefficients
    peaks = find_peaks(coefs.max(axis=0))[0]
    peak_times = peaks / fs
    peak_freqs = freqs[peaks]
    
    return peak_times, peak_freqs

def bayesian_peak_detection(signal, fs):
    # Estimate noise distribution parameters
    noise_mean, noise_std = norm.fit(signal)
    
    # Compute the likelihood of observing the data given the presence or absence of a UAV signal
    likelihood_uav = norm.pdf(signal, loc=noise_mean+3*noise_std, scale=noise_std)
    likelihood_no_uav = norm.pdf(signal, loc=noise_mean, scale=noise_std)
    
    # Compute the posterior probability of a UAV signal being present
    prior_uav = 0.1
    posterior_uav = prior_uav * likelihood_uav / (prior_uav * likelihood_uav + (1-prior_uav) * likelihood_no_uav)
    
    # Identify peaks in the posterior probability
    peaks = find_peaks(posterior_uav, height=0.5)[0]
    peak_times = peaks / fs
    peak_freqs = np.full(len(peaks), np.nan)
    
    return peak_times, peak_freqs

def matched_filtering(signal, fs, template):
    # Normalize the template
    template = template / np.sqrt(np.sum(template**2))
    
    # Compute the cross-correlation between the segment and the template
    corr = np.correlate(signal, template, mode='same')
    
    # Identify peaks in the cross-correlation
    peaks = find_peaks(corr, height=0.5*np.max(corr))[0]
    peak_times = peaks / fs
    peak_freqs = np.full(len(peaks), np.nan)
    
    return peak_times, peak_freqs

input_dir = '/Users/alecchen/Desktop/Documents/Professional/AFRL/distasio/sA1r01/p02'
output_dir = './processed'

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.wav'):
            input_file_path = os.path.join(root, file)
            output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
            os.makedirs(output_subdir, exist_ok=True)
            
            fs, data = wavfile.read(input_file_path)
            fs = int(fs)  # Convert fs to an integer
            segment_length = 30 * fs
            segments = [data[i:i+segment_length] for i in range(0, len(data), segment_length)]
            
            for i, segment in enumerate(segments):
                time = np.arange(len(segment)) / fs
                
                # Wavelet transform-based peak detection
                wavelet_peak_times, wavelet_peak_freqs = wavelet_peak_detection(segment, fs)
                
                # Bayesian peak detection
                bayesian_peak_times, bayesian_peak_freqs = bayesian_peak_detection(segment, fs)
                
                # Matched filtering
                template = np.sin(2*np.pi*1000*np.arange(1000)/fs) * np.exp(-np.arange(1000)/100)
                matched_filter_peak_times, matched_filter_peak_freqs = matched_filtering(segment, fs, template)
                
                # Create dataframes with identified results
                wavelet_peaks_df = pd.DataFrame({'Time (s)': wavelet_peak_times, 'Frequency (Hz)': wavelet_peak_freqs})
                bayesian_peaks_df = pd.DataFrame({'Time (s)': bayesian_peak_times, 'Frequency (Hz)': bayesian_peak_freqs})
                matched_filter_peaks_df = pd.DataFrame({'Time (s)': matched_filter_peak_times, 'Frequency (Hz)': matched_filter_peak_freqs})
                
                # Save dataframes to output directory
                wavelet_peaks_df.to_csv(os.path.join(output_subdir, f'segment_{i+1}_wavelet_peaks.csv'), index=False)
                bayesian_peaks_df.to_csv(os.path.join(output_subdir, f'segment_{i+1}_bayesian_peaks.csv'), index=False)
                matched_filter_peaks_df.to_csv(os.path.join(output_subdir, f'segment_{i+1}_matched_filter_peaks.csv'), index=False)
                
                # Plotting results
                fig, axes = plt.subplots(3, 2, figsize=(12, 12))
                fig.suptitle(f'Segment {i+1}')
                
                axes[0, 0].plot(time, segment)
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].set_title('Time Series')
                
                axes[0, 1].plot(np.fft.fftfreq(len(segment), d=1/fs), np.abs(np.fft.fft(segment)))
                axes[0, 1].set_xlabel('Frequency (Hz)')
                axes[0, 1].set_ylabel('Magnitude')
                axes[0, 1].set_title('Frequency Spectrum')
                
                axes[1, 0].plot(time, coefs.max(axis=0))
                axes[1, 0].plot(wavelet_peak_times, coefs.max(axis=0)[np.round(wavelet_peak_times*fs).astype(int)], 'ro')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Wavelet Coefficient')
                axes[1, 0].set_title('Wavelet Transform-based Peak Detection')
                
                axes[1, 1].plot(time, posterior_uav)
                axes[1, 1].plot(bayesian_peak_times, posterior_uav[np.round(bayesian_peak_times*fs).astype(int)], 'ro')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Posterior Probability')
                axes[1, 1].set_title('Bayesian Peak Detection')
                
                axes[2, 0].plot(time, np.correlate(segment, template, mode='same'))
                axes[2, 0].plot(matched_filter_peak_times, np.correlate(segment, template, mode='same')[np.round(matched_filter_peak_times*fs).astype(int)], 'ro')
                axes[2, 0].set_xlabel('Time (s)')
                axes[2, 0].set_ylabel('Correlation')
                axes[2, 0].set_title('Matched Filtering')
                
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(os.path.join(output_subdir, f'segment_{i+1}_results.png'))
                plt.close(fig)
