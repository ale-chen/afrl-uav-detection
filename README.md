# afrl-uav-detection
Contributors: Marty DiStasio, Alec Chen, Griffin Eychner, Nicholas Adair

**Project is designated as 'Controlled Unclassified'**

## Description

This repository holds the code and data for an SUAS detection project using audio data in a cluttered environment, i.e. crickets, wind, and other background noise.
The code base is split into several sub directories corresponding to the timeline of the project and the tools which contribute to the end system.
Links to the code base can be found below.

### Exploratory Initial
Initial work and findings by Alec to inform and plan the group project.

### Reading
Useful papers, articles, and book exerpts for background reading on the techniques used in this repository.
Includes introductions to audio source separation and non-negative matrix factorization.


### Working Data
Two hand-selected sound files containing drone noise and significant background interference.
Used for small scale testing.

### Noise Subtraction
Includes files and folders for data cleaning and noise subtraction.
 - audioSeparation.ipynb: Provides basic contains basic NMF functionality, ability to plot and save results, and combine specific components
 - averaged_nmf.py: Testing for NMF functionality using generalized matrices from training data (noise_train, noise_train_old), used to produce examples of denoising with different hyperparaters.
Also used to test Harmonic Percussive Sound Separation and its effect on NMF denoising.

### Classify
Files for performing classification tasks and formatting the dataset for those tasks.
 - cnn_test.ipynb: Work in progress for a CNN classifier
 - data_formating.py: Functions to format the dataset for cnn_test.ipynb

### Writeup
Slide deck to track project progress.


## Dependencies
This project uses
 - PyTorch
 - Pandas
 - sklearn
 - Librosa
 - soundfile
 - tqdm
 - rich

## Links

**This is costing Alec (and possibly also Griffin) money to store the data online**
[Link to Data](https://drive.google.com/drive/folders/1N5liu26akYoOCsA3PL5ZrG9DrWhQlS37?usp=drive_link)  

[Link to segmented data](https://drive.google.com/drive/folders/14sl4ynlxHpKeUJ1dgy56ZY_OcQ2I64Hl?usp=drive_link)
