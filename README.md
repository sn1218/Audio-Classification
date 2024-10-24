# Audio Classification using a Convolution Neural Network
## Overview

This project focuses on classifying songs into their respective genres based on audio data, drawing inspiration from platforms like Spotify that categorise music. Using a Convolutional Neural Network (CNN) model, the primary objective was to surpass the accuracy of random guessing (10% for 10 genres). The project leverages the widely-used [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), with core tools including Python libraries like librosa for audio feature extraction and tensorflow for building and training the CNN model.


## Project Details
### Tools and Libraries
This project was implemented in Python and utilised several key libraries, including:

* **pandas and numpy:** For data manipulation
* **matplotlib and seaborn:** For visualising the data
* **librosa:** For extracting audio features such as MFCCs, Mel spectrograms, chroma vectors, and tonnetz
* **tensorflow:** For constructing and training the neural network

### Dataset
The project uses the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which contains 1,000 audio files, each 30 seconds long, classified into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

The data is sourced from Kaggle using the Kaggle API.

#### Data Preprocessing
To prepare the data for training, the following steps were performed:

* **Data cleaning:** Corrupted or unreadable files were removed to ensure a clean dataset.
* **Feature extraction:** Multiple audio features were extracted using librosa, including:
  * MFCCs (Mel-frequency cepstral coefficients)
  * Mel spectrograms
  * Chroma vectors
  * Tonnetz features (tonal centroid features)
* **Feature summarisation:** For each extracted feature, statistics such as minimum, maximum, and mean were computed. These were used as inputs to the CNN model.

### Model Architeture 
This CNN model classifies audio tracks into genres by processing the extracted features. Here's a high-level overview of the layers:

* **Convolutional Layer 1:** Detects basic audio features from the inputs.
* **Max Pooling Layer 1:** Reduces the size of the feature map to retain important information and reduce complexity.
* **Dropout Layer:** Reduces overfitting by randomly deactivating some neurons during training.
* **Convolutional Layer 2:** Learns more complex patterns from the audio data.
* **Max Pooling Layer 2:** Further reduces the size of the feature map.
* **Dropout Layer:** Adds further regularisation to improve model generalisation.
* **Convolutional Layer 3:** Captures even higher-level patterns in the audio features.
* **Max Pooling Layer 3:** Reduces dimensionality before flattening.
* **Dropout Layer:** Another layer to prevent overfitting.
* **Flatten Layer:** Converts the 2D feature maps into a 1D vector to prepare for the dense (fully connected) layers.
* **Dense Layer:** A fully connected layer that learns the relationship between the patterns and genres.
* **Dropout Layer:** Final regularisation before the output.
* **Output Layer:** Classifies the track into one of the 10 genre categories using a softmax activation function.

  ![Model Architecture](https://github.com/user-attachments/assets/a5bff676-8fbe-4e7a-9c9e-d0299a0c2a6f)


### Results
The CNN model achieved a 65% accuracy on the test dataset, significantly outperforming the random guessing baseline of 10%. The model performed particularly well in identifying classical music, which may be attributed to the distinct characteristics of this genre.

## Repository Structure
* **Audio_Data_EDA.ipynb:** Jupyter notebook for Exploratory Data Analysis (EDA) of the GTZAN dataset.
* **Classification_of_Audio_Data_using_Machine_Learning.ipynb:** Main notebook detailing the CNN model development, training, and evaluation process.
* **Test_Model_on_Youtube_Clips.ipynb:** Notebook for testing the trained model on new audio samples sourced from YouTube.
* **cnn_model.h5:** Pre-trained CNN model file.
* **history.pkl:** Contains the model's training history.
* **classes.npy:** Numpy array storing the genre label encodings used in the model.
* **feature_extraction.py:** Script for extracting audio features for the CNN model.

### Project Presentation
You can view the animated slide deck that summarises this project [here](https://www.canva.com/design/DAGJ_ayphRI/afFF12HA3axhxRTSc2otRA/view?utm_content=DAGJ_ayphRI&utm_campaign=designshare&utm_medium=link&utm_source=editor). The presentation outlines the motivation, methodology, and results of the project. A static PDF version is also available in the repository.

## How to Use the Project
1. Clone the repository and download the following files:
  * feature_extraction.py
  * classes.npy
  * cnn_model.h5
  * Test_Model_on_Youtube_Clips.ipynb
2. Follow the steps in Test_Model_on_Youtube_Clips.ipynb to load the model and test it using audio clips from YouTube.
