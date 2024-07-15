import numpy as np
import librosa

def extract_features(file_path):

  y, sr = librosa.load(file_path)

  # Extracting MFCC feature
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

  mfcc_mean = np.mean(mfcc, axis=1)
  mfcc_min = np.min(mfcc, axis=1)
  mfcc_max = np.max(mfcc, axis=1)
  mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

  # Extracting Mel Spectrogram feature
  melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
  melspectrogram_mean = np.mean(melspectrogram, axis=1)
  melspectrogram_min = np.min(melspectrogram, axis=1)
  melspectrogram_max = np.max(melspectrogram, axis=1)
  melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

  # Extracting chroma vector feature
  chroma = librosa.feature.chroma_stft(y=y, sr=sr)
  chroma_mean = np.mean(chroma, axis=1)
  chroma_min = np.min(chroma, axis=1)
  chroma_max = np.max(chroma, axis=1)
  chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

  # Extracting tonnetz feature
  tntz = librosa.feature.tonnetz(y=y, sr=sr)
  tntz_mean = np.mean(tntz, axis=1)
  tntz_min = np.min(tntz, axis=1)
  tntz_max = np.max(tntz, axis=1)
  tntz_feature = np.concatenate( (tntz_mean, tntz_min, tntz_max) )

  feature = np.concatenate( (mfcc_feature, melspectrogram_feature, chroma_feature, tntz_feature) )

  feature = np.array(feature)
  feature = np.expand_dims(feature, axis=-1)  # Add a channel dimension for CNN input
  feature = np.expand_dims(feature, axis=0)

  return feature
