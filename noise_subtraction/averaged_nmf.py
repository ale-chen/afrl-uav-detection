import librosa
import numpy as np
from sklearn.decomposition import NMF

class AudioDenoiser:
    def __init__(self, noise_files, n_components=10, n_fft=1024, hop_length=512):
        self.noise_files = noise_files
        self.n_components = n_components
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.basis_matrices = []
        self.nmf_models = []

    def preprocess_audio(self, file_path):
        audio, sr = librosa.load(file_path)
        spectrogram = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(spectrogram)
        return magnitude

    def train_nmf(self, spectrogram):
        nmf = NMF(n_components=self.n_components, random_state=0, max_iter=1000)
        basis_matrix = nmf.fit_transform(spectrogram)
        return basis_matrix, nmf.components_

    def train(self):
        for noise_files in self.noise_files:
            basis_matrices = []
            for file in noise_files:
                noise_spectrogram = self.preprocess_audio(file)
                basis_matrix, _ = self.train_nmf(noise_spectrogram)
                basis_matrices.append(basis_matrix)
            
            averaged_basis_matrix = np.mean(basis_matrices, axis=0)
            self.basis_matrices.append(averaged_basis_matrix)

            nmf = NMF(n_components=self.n_components, init='custom', solver='cd', max_iter=1000)
            nmf.components_ = averaged_basis_matrix.T
            self.nmf_models.append(nmf)

    def denoise_audio(self, spectrogram):
        denoised_spectrogram = spectrogram.copy()
        for nmf, basis_matrix in zip(self.nmf_models, self.basis_matrices):
            activation_matrix = nmf.transform(denoised_spectrogram.T)
            noise_spectrogram = np.dot(basis_matrix, activation_matrix).T
            noise_spectrogram = librosa.util.fix_length(noise_spectrogram, denoised_spectrogram.shape[1], axis=1)
            denoised_spectrogram -= noise_spectrogram
        return denoised_spectrogram

    def reconstruct_audio(self, denoised_spectrogram, phase):
        complex_spectrogram = denoised_spectrogram * np.exp(1j * phase)
        denoised_audio = librosa.istft(complex_spectrogram, hop_length=self.hop_length)
        return denoised_audio

    def denoise(self, audio_file, output_file):
        input_spectrogram = self.preprocess_audio(audio_file)
        denoised_spectrogram = self.denoise_audio(input_spectrogram)
        phase = np.angle(librosa.stft(librosa.load(audio_file)[0], n_fft=self.n_fft, hop_length=self.hop_length))
        denoised_audio = self.reconstruct_audio(denoised_spectrogram, phase)
        librosa.output.write_wav(output_file, denoised_audio, sr=22050)

# Usage example
if __name__ == '__main__':
    noise_files_1 = ['noise_train/crickets_1.wav', 'noise_train/crickets_2.wav', 'noise_train/crickets_3.wav', 'noise_train/crickets_4.wav']
    noise_files_2 = ['noise_train/wind_1.wav', 'noise_train/wind_2.wav']
    noise_files_3 = ['noise_train/crickets_speaking.wav']

    audio_file = '../working_data/d303sA1r01p0120210823.wav'
    output_file = 'results/denoised_audio.wav'
    
    denoiser = AudioDenoiser(noise_files=[noise_files_1, noise_files_2, noise_files_3], n_components=10)
    denoiser.train()
    denoiser.denoise(audio_file, output_file)

