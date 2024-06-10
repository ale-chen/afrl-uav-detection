import librosa
import numpy as np
from sklearn.decomposition import NMF


class AudioDenoiser:
    """
    A class for audio denoising using Non-Negative Matrix Factorization (NMF).

    Attributes:
        n_components (int): Number of components for NMF decomposition.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        noise_files_list (list): List of noise file paths.
        noise_type_matrices (list): List of generalized noise type matrices.
    """

    def __init__(self, n_components=10, n_fft=1024, hop_length=512):
        """
        Initialize the AudioDenoiser.

        Args:
            n_components (int, optional): Number of components for NMF decomposition. Defaults to 10.
            n_fft (int, optional): FFT window size. Defaults to 1024.
            hop_length (int, optional): Hop length for STFT. Defaults to 512.
        """
        self.n_components = n_components
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_files_list = []
        self.noise_type_matrices = []

    def compute_basis_matrix(self, audio_file, max_iter = 1200):
        """
        Compute the basis and activation matrices for an audio file using NMF.

        Args:
            audio_file (str): Path to the audio file.
            max_iter (int, optional): Maximum number of iterations for NMF. Defaults to 1200.


        Returns:
            tuple: A tuple containing the basis matrix (W) and the activation matrix (H).
        """
        y, sr = librosa.load(audio_file)
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag, _ = librosa.magphase(S)

        nmf = NMF(n_components=self.n_components, max_iter = max_iter, random_state=0)
        W = nmf.fit_transform(S_mag)
        H = nmf.components_

        return W, H

    def generalize_noise_types(self, noise_files_list):
        """
        Generalize the noise types by computing the average basis matrix for each noise type.

        Args:
            noise_files_list (list): List of lists, where each inner list contains noise file paths for a specific noise type.

        Returns:
            list: List of generalized noise type matrices.
        """
        self.noise_files_list = noise_files_list
        self.noise_type_matrices = []

        for noise_files in noise_files_list:
            noise_type_basis_matrices = []

            for noise_file in noise_files:
                W, _ = self.compute_basis_matrix(noise_file)
                noise_type_basis_matrices.append(W)

            noise_type_matrix = np.mean(noise_type_basis_matrices, axis=0)
            self.noise_type_matrices.append(noise_type_matrix)

        return self.noise_type_matrices


# Example Usage

noise_files_1 = ['noise_train/crickets_1.wav', 'noise_train/crickets_2.wav', 'noise_train/crickets_3.wav', 'noise_train/crickets_4.wav']
noise_files_2 = ['noise_train/wind_1.wav', 'noise_train/wind_2.wav']
noise_files_3 = ['noise_train/crickets_speaking.wav']

denoiser = AudioDenoiser()
noise_type_matrices = denoiser.generalize_noise_types([noise_files_1, noise_files_2, noise_files_3])

print(noise_type_matrices)