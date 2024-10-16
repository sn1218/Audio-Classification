# Audio Classification
## Overview

This project classifies songs into their respective genres using audio data, inspired by how platforms like Spotify categorise music. By leveraging a Convolutional Neural Network (CNN) model, the goal was to achieve a classification accuracy higher than random guessing (10% for 10 genres). The project uses the GTZAN dataset, and key tools include Python libraries like librosa for audio feature extraction and tensorflow for building the CNN model.


## The Project

The tools used in this project include: Python and Python libraries (pandas, numpy, matplotlib, librosa, tensorflow).

### Dataset
The project uses the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which contains 1,000 audio files, each 30 seconds long, classified into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

#### Data Preprocessing
* Files that were corrupted or failed to load properly were removed.
* Mel spectrograms were extracted using librosa, and features such as min, max, and mean of each row were computed for input into the CNN model.
* Final features were used to train the CNN model.

### Results
The CNN model achieved an accuracy of 65% on the test set, significantly outperforming the random guessing baseline of 10%.

The model best identifies classical music.

### Key Files in the Repository
* Audio_Data_EDA.ipynb: Performs exploratory data analysis (EDA) on the dataset.
* Classification_of_Audio_Data_using_Machine_Learning.ipynb: Contains the steps for building and training the CNN model.
* Test_Model_on_Youtube_Clips.ipynb: Tests the trained model on new audio samples from YouTube.
* cnn_model.h5: The saved CNN model.
* history.pkl: The training history of the model.
* classes.npy: The label encoding used in the model.
* feature_extraction.py: Module for extracting audio features for model input.

You can view the animated slide deck that summarises this project [here](https://www.canva.com/design/DAGJ_ayphRI/afFF12HA3axhxRTSc2otRA/view?utm_content=DAGJ_ayphRI&utm_campaign=designshare&utm_medium=link&utm_source=editor). The presentation outlines the motivation, methodology, and results of the project. A static PDF version is also available in the repository.
