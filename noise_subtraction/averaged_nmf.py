import librosa
import soundfile as sf
import numpy as np
from sklearn.decomposition import NMF
# from sklearn.metrics  import mean_squared_error
import os
from multiprocessing import Pool


class CustomNMF(NMF):
    """
    A custom implementation of Non-Negative Matrix Factorization (NMF) that allows for fixed components.

    This class extends the NMF class from sklearn.decomposition and adds the capability to fix a subset of components,
    while the remaining components are updated during the factorization process.

    Attributes:
        fixed_components (numpy.ndarray): An optional matrix containing fixed components. If provided,
            the factorization will be performed with these fixed components and additional randomly initialized components.

    All other attributes are inherited from the sklearn.decomposition.NMF class.

    Methods:
        fit_transform(X, y=None, W=None, H=None): Compute the non-negative matrix factorization of the provided data.
            If fixed_components are provided, they are used as part of the initialization, and only the remaining
            components are updated during the factorization process.
        _update_H(X, W, H): Update the activation matrix H using the multiplicative update rule.
        _update_W(X, W, H): Update the basis matrix W using the multiplicative update rule.

    All other methods are inherited from the sklearn.decomposition.NMF class.
    """

    def __init__(self, n_components=None, *, init=None, solver='cd', beta_loss='frobenius',
                 tol=1e-4, max_iter=2048, random_state=None, alpha_W=0., alpha_H=0.,
                 l1_ratio=0., verbose=0, shuffle=False, fixed_components=None):
        super().__init__(
            n_components=n_components, init=init, solver=solver, beta_loss=beta_loss,
            tol=tol, max_iter=max_iter, random_state=random_state, alpha_W=alpha_W,
            alpha_H=alpha_H, l1_ratio=l1_ratio, verbose=verbose, shuffle=shuffle)
        self.fixed_components = fixed_components

    def fit_transform(self, X, y=None, W=None, H=None):
        """
        Compute the non-negative matrix factorization of the provided data.

        Args:
            X (numpy.ndarray): Input data matrix.
            y (Ignored): Not used, present for API consistency.
            W (numpy.ndarray, optional): If not None, the initial value of the basis matrix.
            H (numpy.ndarray, optional): If not None, the initial value of the activation matrix.

        Returns:
            tuple: (W, H), where W is the basis matrix, and H is the activation matrix.
        """ 
        if self.fixed_components is not None:
            W_fixed = self.fixed_components
            n_fixed_components = W_fixed.shape[1]
            n_update_components = self.n_components - n_fixed_components

            # Initialize W with fixed and random components
            W_update = np.random.rand(X.shape[0], n_update_components)
            W = np.hstack((W_fixed, W_update))

            # Initialize H with random values
            H = np.random.rand(self.n_components, X.shape[1])

            # Optimize the signal components in W and the entire H matrix
            for i in range(self.max_iter):
                # Update H
                H = self._update_H(X, W, H)

                # Update the signal components in W
                W[:, n_fixed_components:] = self._update_W(X, W, H)[:, n_fixed_components:]

                # Track fit progress with Euclidean norm
                # if i % 100 == 0:
                #     print(f"Index {i}:\n {mean_squared_error(X, W @ H)}")

            self.components_ = H
        else:
            W = super().fit_transform(X, W=W, H=H)
            H = self.components_

        return W, H

    def _update_H(self, X, W, H):
        """
        Update the activation matrix H using the multiplicative update rule.

        Args:
            X (numpy.ndarray): Input data matrix.
            W (numpy.ndarray): Basis matrix.
            H (numpy.ndarray): Activation matrix.

        Returns:
            numpy.ndarray: Updated activation matrix.
        """
        numerator = W.T @ X
        denominator = W.T @ (W @ H)
        H *= numerator / denominator
        return H

    def _update_W(self, X, W, H):
        """
        Update the basis matrix W using the multiplicative update rule.

        Args:
            X (numpy.ndarray): Input data matrix.
            W (numpy.ndarray): Basis matrix.
            H (numpy.ndarray): Activation matrix.

        Returns:
            numpy.ndarray: Updated basis matrix.
        """
        numerator = X @ H.T
        denominator = W @ (H @ H.T)
        W *= numerator / denominator
        return W


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

    def __init__(self, n_components=10, n_fft=1024, hop_length=768,  load_noise_type_matrices=False):
        """
        Initialize the AudioDenoiser.

        Args:
            n_components (int, optional): Number of components for NMF decomposition. Defaults to 10.
            n_fft (int, optional): FFT window size. Defaults to 1024.
            hop_length (int, optional): Hop length for STFT. Defaults to 512.
            load_noist_type_matrices (bool, optional): Flag to load the generalized noise type matrices from memory.
        """
        self.n_components = n_components
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.load_noise_type_matrices = load_noise_type_matrices

        self.matrix_directory = "./noise_type_matrices"
        self.noise_files_list = []
        self.noise_type_matrices = []

    def compute_basis_matrix(self, audio_file, max_iter = 2048):
        """
        Compute the basis and activation matrices for an audio file using NMF.

        Args:
            audio_file (str): Path to the audio file.
            max_iter (int, optional): Maximum number of iterations for NMF. Defaults to 1200.


        Returns:
            tuple: A tuple containing the basis matrix (W) and the activation matrix (H).
        """
        y, sr = librosa.load(audio_file)
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length) # This single line can be changed for omd if we would like.
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

        if self.load_noise_type_matrices:
            return self.load_noise_type_matrices(self.matrix_directory)
        
        self.noise_files_list = noise_files_list
        self.noise_type_matrices = []

        for noise_files in noise_files_list:
            noise_type_basis_matrices = []

            for noise_file in noise_files:
                W, _ = self.compute_basis_matrix(noise_file)
                noise_type_basis_matrices.append(W)

            noise_type_matrix = np.mean(noise_type_basis_matrices, axis=0)
            self.noise_type_matrices.append(noise_type_matrix)

        # Save noise type matrices
        if not self.load_noise_type_matrices:
            self.save_noise_type_matrices(self.matrix_directory)

        return self.noise_type_matrices
    
    def save_noise_type_matrices(self, output_dir):
        """
        Save the generalized noise type matrices to the specified output directory.

        Args:
            output_dir (str): Directory to save the generalized noise type matrices.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, noise_type_matrix in enumerate(self.noise_type_matrices):
            output_path = os.path.join(output_dir, f"noise_type_{i+1}.npy")
            np.save(output_path, noise_type_matrix)

    def load_noise_type_matrices(self, input_dir):
        """
        Load the generalized noise type matrices from the specified input directory.

        Args:
            input_dir (str): Directory to load the generalized noise type matrices.
        """
        if not os.path.exists(input_dir):
            raise Exception(f"Directory {input_dir} does not exist.")
        
        noise_type_matrices = []

        for path in os.listdir(input_dir):
            if os.is_file(path):
                noise_type_matrices.append(np.load(path))

        return noise_type_matrices

    def denoise_audio(self, audio_file, n_signal_components=2, max_iter=2048):
        """
        Denoise an audio file using Non-Negative Matrix Factorization (NMF) with fixed noise basis vectors.

        Args:
            audio_file (str): Path to the audio file to be denoised.
            n_signal_components (int, optional): Number of signal components to be used in the NMF decomposition.
                Defaults to 2.
            max_iter (int, optional): Maximum number of iterations for NMF. Defaults to 1200.

        Returns:
            tuple: A tuple containing the following:
                - signal_basis_vectors (numpy.ndarray): The basis vectors representing the signal components.
                - signal_activations (numpy.ndarray): The activation matrix for the signal components.
                - denoised_file (str): Path to the denoised audio file.
        """
        # Load the audio file and compute the magnitude spectrogram
        y, sr = librosa.load(audio_file)
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag, _ = librosa.magphase(S)

        # Stack the generalized noise type matrices horizontally
        noise_basis_matrix = np.hstack(self.noise_type_matrices)

        # Initialize the signal basis matrix randomly
        signal_basis_matrix = np.random.rand(S_mag.shape[0], n_signal_components)

        # Concatenate the noise and signal basis matrices
        initial_basis_matrix = np.hstack((noise_basis_matrix, signal_basis_matrix))

        # Perform NMF with fixed noise basis vectors
        nmf = CustomNMF(n_components=initial_basis_matrix.shape[1], max_iter=max_iter, random_state=0,
                        fixed_components=noise_basis_matrix)
        W, H = nmf.fit_transform(S_mag, W=initial_basis_matrix, H=None)

        # Extract the signal basis vectors and corresponding activations
        signal_basis_vectors = W[:, noise_basis_matrix.shape[1]:]
        signal_activations = H[noise_basis_matrix.shape[1]:, :]

        # Reconstruct the denoised spectrogram
        denoised_S_mag = np.dot(signal_basis_vectors, signal_activations)
        denoised_S = denoised_S_mag * np.exp(1j * np.angle(S))

        # Synthesize the denoised audio
        denoised_audio = librosa.istft(denoised_S, hop_length=self.hop_length)

        # Save the denoised audio to a file
        denoised_file = os.path.splitext(audio_file)[0] + "_denoised.wav"
        sf.write(denoised_file, denoised_audio, samplerate=sr, subtype='PCM_24')

        return signal_basis_vectors, signal_activations, denoised_file
    
    def denoise_audio_parallel(self, audio_files, n_signal_components=2, max_iter=2048, n_processes=None):
        """
        Denoise multiple audio files in parallel using Non-Negative Matrix Factorization (NMF) with fixed noise basis vectors.

        Args:
            audio_files (list): List of paths to the audio files to be denoised.
            n_signal_components (int, optional): Number of signal components to be used in the NMF decomposition.
                Defaults to 2.
            max_iter (int, optional): Maximum number of iterations for NMF. Defaults to 1200.
            n_processes (int, optional): Number of processes to use for parallel processing. If None, it uses the number of CPUs available.

        Returns:
            list: A list of tuples, where each tuple contains the following:
                - signal_basis_vectors (numpy.ndarray): The basis vectors representing the signal components.
                - signal_activations (numpy.ndarray): The activation matrix for the signal components.
                - denoised_file (str): Path to the denoised audio file.
        """
        with Pool(processes=n_processes) as pool:
            results = pool.starmap(self.denoise_audio, [(audio_file, n_signal_components, max_iter) for audio_file in audio_files])

        return results
    

# Example Usage
def example():
    noise_files_1 = []
    cricket_path = "./noise_train/crickets"
    for path in os.listdir(cricket_path):
                noise_files_1.append(cricket_path + '/' + path)
    # noise_files_1 = ['noise_train_old/crickets_1.wav', 'noise_train_old/crickets_2.wav', 'noise_train_old/crickets_3.wav', 'noise_train_old/crickets_4.wav']
    noise_files_2 = ['noise_train_old/wind_1.wav', 'noise_train_old/wind_2.wav']
    noise_files_3 = ['noise_train_old/crickets_speaking.wav']

    denoiser = AudioDenoiser(n_components=8, hop_length=768, load_noise_type_matrices=False)
    noise_type_matrices = denoiser.generalize_noise_types([noise_files_1, noise_files_2, noise_files_3])

    # Denoise multiple signal+noise .wav files in parallel
    signal_noise_files = ['../working_data/d302sA1r01p0120210823.wav', '../working_data/d303sA1r01p0120210823.wav']
    denoised_results = denoiser.denoise_audio_parallel(signal_noise_files, n_signal_components=8)

    for result in denoised_results:
        denoised_basis, denoised_activations, denoised_file = result
        print("Denoised basis matrix shape:", denoised_basis.shape)
        print("Denoised activation matrix shape:", denoised_activations.shape)
        print("Denoised audio file:", denoised_file)


# Finding the best hyperparameter values (n_components, n_fft, hop_length, n_signal_components, max_iter)
def hyper_parameter_test():
    n_components = 1024
    max_iter = 2048
    hop_length = 768
    print(f"Components: {n_components}")
    print(f"Iterations: {max_iter}")
    print(f"Hop Length: {hop_length}")

    # Load the audio file and compute the magnitude spectrogram
    y, sr = librosa.load('noise_train/crickets_1.wav')
    print(f"Sampling Rate:{sr}")
    S = librosa.stft(y, n_fft=1024, hop_length=hop_length)
    S_mag, _ = librosa.magphase(S)

    # Perform NMF with fixed noise basis vectors
    nmf = CustomNMF(n_components=n_components, max_iter=max_iter, random_state=0,
                    fixed_components=None)
    W, H = nmf.fit_transform(S_mag, W=None, H=None)

    # Reconstruct the denoised spectrogram
    denoised_S_mag = np.dot(W, H)
    denoised_S = denoised_S_mag * np.exp(1j * np.angle(S))

    # Synthesize the denoised audio
    denoised_audio = librosa.istft(denoised_S, hop_length=hop_length)

    # Save the denoised audio to a file
    denoised_file = os.path.splitext('noise_train/crickets_1.wav')[0] + "_processed.wav"
    sf.write(denoised_file, denoised_audio, samplerate=sr, subtype='PCM_24')

    # Denoise a signal+noise .wav file
    # signal_noise_file = "../working_data/d303sA1r01p0120210823.wav"
    # denoiser.denoise_audio(signal_noise_file, n_signal_components=2, max_iter=2048)


if __name__ ==  "__main__":
    # hyper_parameter_test()
    example()