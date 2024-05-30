import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, freqz
import os

def ideal_bandpass(lowcut, highcut, fs, n):
    freq = np.fft.fftfreq(n, d=1/fs)
    mask = np.logical_or(np.logical_and(freq >= lowcut, freq <= highcut),
                         np.logical_and(freq <= -lowcut, freq >= -highcut))
    h = np.zeros(n, dtype=np.complex128)
    h[mask] = 1
    h = np.fft.ifft(h)
    h = np.fft.fftshift(h)
    h = np.real(h)
    return h

def ideal_bandpass_filter(data, lowcut, highcut, fs, n):
    h = ideal_bandpass(lowcut, highcut, fs, n)
    y = np.convolve(data, h, mode='same')
    return y, h

input_dir = '/Users/alecchen/Desktop/Documents/Professional/AFRL/distasio/sA1r01/p02'
output_dir = './processed'

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.wav'):
            input_file_path = os.path.join(root, file)
            output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
            os.makedirs(output_subdir, exist_ok=True)

            fs, data = wavfile.read(input_file_path)

            freq_ranges = [(0, 3000), (3000, 6000), (6000, 9000), (9000, fs//2)]

            segment_length = 30 * fs
            segments = [data[i:i+segment_length] for i in range(0, len(data), segment_length)]

            for i, segment in enumerate(segments):
                for j, (lowcut, highcut) in enumerate(freq_ranges):
                    n = len(segment)
                    filtered_segment, h = ideal_bandpass_filter(segment, lowcut, highcut, fs, n)

                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    fig.suptitle(f'Segment {i+1}, Bandpass {lowcut}-{highcut} Hz')

                    axes[0, 0].plot(np.arange(len(data))/fs, data)
                    axes[0, 0].set_xlabel('Time (s)')
                    axes[0, 0].set_ylabel('Amplitude')
                    axes[0, 0].set_title('Full Timeseries')

                    axes[0, 1].plot(np.arange(len(segment))/fs, segment)
                    axes[0, 1].set_xlabel('Time (s)')
                    axes[0, 1].set_ylabel('Amplitude')
                    axes[0, 1].set_title(f'Segment {i+1} Timeseries')
                    
                    freq_hz = np.fft.fftfreq(len(h), d=1/fs)
                    magnitude = np.abs(np.fft.fft(h))

                    axes[1, 0].plot(freq_hz, magnitude)
                    axes[1, 0].set_xlabel('Frequency (Hz)')
                    axes[1, 0].set_ylabel('Magnitude')
                    axes[1, 0].set_title(f'Ideal Bandpass Filter ({lowcut}-{highcut} Hz)')
                    axes[1, 0].set_xlim(0, fs//2)
                    axes[1, 0].set_ylim(0, 1.1)
                    axes[1, 0].axvline(lowcut, color='r', linestyle='--', linewidth=1)
                    axes[1, 0].axvline(highcut, color='r', linestyle='--', linewidth=1)


                    axes[1, 1].plot(np.arange(len(filtered_segment))/fs, filtered_segment)
                    axes[1, 1].set_xlabel('Time (s)')
                    axes[1, 1].set_ylabel('Amplitude')
                    axes[1, 1].set_title(f'Filtered Segment {i+1}')

                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.savefig(os.path.join(output_subdir, f'segment_{i+1}_bandpass_{lowcut}-{highcut}.png'))
                    plt.close(fig)
