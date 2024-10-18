# Audio Classification
## Overview

This project focuses on classifying songs into their respective genres based on audio data, drawing inspiration from platforms like Spotify that categorise music. Using a Convolutional Neural Network (CNN) model, the primary objective was to surpass the accuracy of random guessing (10% for 10 genres). The project leverages the widely-used [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), with core tools including Python libraries like librosa for audio feature extraction and tensorflow for building and training the CNN model.


## Project Details
### Tools and Libraries
This project was implemented in Python and utilised several key libraries, including:

* pandas and numpy for data manipulation
* matplotlib and seaborn for visualisation
* librosa for audio processing and feature extraction
* tensorflow for constructing and training the neural network

### Dataset
The project uses the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which contains 1,000 audio files, each 30 seconds long, classified into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

#### Data Preprocessing
To prepare the dataset for the CNN model:

* Data cleaning: Corrupted or unreadable files were removed to ensure a clean dataset.
* Feature extraction: MFCCs, Mel spectrograms, chroma vectors, and tonnetz features were extracted using librosa.
* Feature summarisation: Statistical features such as minimum, maximum, and mean values were calculated from the extracted features and used as input for the CNN model.

### Model Architeture 
This CNN model is designed to classify audio tracks into genres based on features extracted from the MFCC, Mel spectogram, chroma vector, and tonnetz feature graphs. Here's an overview of the layers:

* Convolutional Layer 1: Detects basic audio features from the Mel spectrogram input.
* Max Pooling Layer 1: Reduces the size of the feature map to retain important information and reduce complexity.
* Dropout Layer: Helps prevent overfitting by randomly disabling a fraction of neurons during training.
* Convolutional Layer 2: Learns more complex patterns from the audio data.
* Max Pooling Layer 2: Further reduces the size of the feature map.
* Dropout Layer: Adds further regularization to improve model generalization.
* Convolutional Layer 3: Captures even higher-level patterns in the audio features.
* Max Pooling Layer 3: Reduces dimensionality before flattening.
* Dropout Layer: Another layer to prevent overfitting.
* Flatten Layer: Converts the 2D feature maps into a 1D vector to prepare for the dense (fully connected) layers.
* Dense Layer: A fully connected layer that uses the learned patterns for classification.
* Dropout Layer: Final regularization before the output.
* Output Layer: Classifies the audio track into one of the 10 genre categories using a softmax function.

  

### Results
The CNN model achieved a 65% accuracy on the test dataset, significantly outperforming the random guessing baseline of 10%. The model performed particularly well in identifying classical music, which may be attributed to the distinct characteristics of this genre.

### Repository Structure
* Audio_Data_EDA.ipynb: Jupyter notebook for Exploratory Data Analysis (EDA) of the GTZAN dataset.
* Classification_of_Audio_Data_using_Machine_Learning.ipynb: Main notebook detailing the CNN model development, training, and evaluation process.
* Test_Model_on_Youtube_Clips.ipynb: Notebook for testing the trained model on new audio samples sourced from YouTube.
* cnn_model.h5: Pre-trained CNN model file.
* history.pkl: Contains the model's training history.
* classes.npy: Numpy array storing the genre label encodings used in the model.
* feature_extraction.py: Script for extracting and processing audio features used as input for the CNN model.

## Project Presentation
You can view the animated slide deck that summarises this project [here](https://www.canva.com/design/DAGJ_ayphRI/afFF12HA3axhxRTSc2otRA/view?utm_content=DAGJ_ayphRI&utm_campaign=designshare&utm_medium=link&utm_source=editor). The presentation outlines the motivation, methodology, and results of the project. A static PDF version is also available in the repository.
