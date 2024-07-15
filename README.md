# Audio Classification
## The Scenario

My task for this project was to classify a song's genre using its audio data. My aim was to get a higher accuracy than if I were to guess the genre at random; if the number of genres was 10, then the target metric to beat was 10%.

This project was inspired by thoughts on how Spotify classifies their music. 


## The Project

The tools used in this project include: Python and Python libraries (pandas, numpy, matplotlib, librosa, tensorflow).

Audio data was sourced from the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) on Kaggle. Files that would not load properly were dropped, and exploratory data analysis was performed in 'Audio_Data_EDA.ipynb'. 

'Classification_of_Audio_Data_using_Machine_Learning.ipynb' contains all the steps taken to produce the Convolutional Neural Network (CNN) model to classify audio into music genres. The final model was saved as 'cnn_model.h5'; its history is saved as 'history.pkl'. The label encoding is saved as 'classes.npy'. 'feature_extraction.py'

The model and associated files, as well as the module 'feature_extraction.py', are loaded in 'Test_Model_on_Youtube_Clips.ipynb'. 'feature_extraction.py' produces the features used in the CNN model, e.g. min, max, and mean of each row of the Mel Spectogram. 

The animated slide deck presentation can be found [here](https://www.canva.com/design/DAGJ_ayphRI/afFF12HA3axhxRTSc2otRA/view?utm_content=DAGJ_ayphRI&utm_campaign=designshare&utm_medium=link&utm_source=editor). The PDF'd version is included in the repository.
