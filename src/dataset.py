import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

TESS_EMOTIONS = ["angry", "disgust", "fear", "happy", "pleasant_surprise", "sad", "neutral"]
RAVDESS_EMOTIONS = ["neutral", "calm", "happiness", "sadness", "angry", "fearful", "diisgust", "surprised"]

def load_mel_spectrogram(file_path, sr=16000, n_mels=128, max_length=128):
    """
    Load a .wav file and convert it to 3-channels Mel-spectrogram (db scale), to get a tensor
    with shape (3, n_mels, max_length) which is compatible with CNNs
    """

    y, sr = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # pad if too short, crop if too long
    if mel_db.shape[1] < max_length:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_length - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :max_length]

    
    # Reapeat grayscale -> 3 channels 
    mel_db = np.repeat(mel_db[np.newaxis, :, :], 3, axis=0)
    return mel_db.astype(np.float32)

class TESSDataset(Dataset):
    """
    Toronto Emotional Speech Set (TESS).
    Folder structure: TESS/YAF_<emotion>/ and TESS/OAF_<emotion>/
    Emotion is extracted from the folder name.
    """
    def __init__(self, data_path):
        self.samples = []
        self.labels = []
        self.emotion2idx = {e: i for i, e in enumerate(TESS_EMOTIONS)}

        # Extract samples (from audio .wav) and labels for each emotion
        for folder in os.listdir(data_path):
            foler_path = os.path.join(data_path, folder)
            if not os.path.isdir(foler_path):
                continue

            emotion = folder.split("_")[1:].lower()

            if emotion not in self.emotion2idx:
                continue
            for fname in os.listdir(foler_path):
                if fname.endswith(".wav"):
                    self.samples.append(os.path.join(foler_path, fname))
                    self.labels.append(self.emotion2idx[emotion])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel = load_mel_spectrogram(self.samples[idx])
        label = self.labels[idx]
        return torch.tensor(mel), torch.tensor(label) #  mel -> (3, n_mels, max_length)

    
class RAVDESSDataset(Dataset):
    """
    Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
    Filename format: 03-01-06-01-02-01-12.wav
    3rd segment encodes emotion: 01=neutral, 02=calm, 03=happy, ...
    """
    def __intit__(self, data_path):
        self.samples = []
        self.labels = []
        self.emotions2idx = {e: i for i, e in enumerate(RAVDESS_EMOTIONS)} # for simple indexing

        for actor_folder in sorted(os.listdir(data_path)): 
            actor_path = os.path.join(data_path, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            for fname in os.listdir(actor_folder):
                if not fname.endswith(".wav"):
                    continue
                parts = fname.split("-")
                if len(parts) < 3: # because emotions are at the 3rd segment
                    continue

                emotion_id = int(parts[2]) - 1 # RAVDESS uses 1-indexed so we need to reindex with 0-indexed

                if 0 <= emotion_id < len(RAVDESS_EMOTIONS):
                    self.samples.append(os.path.join(data_path, fname))
                    self.labels.append(emotion_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mel = load_mel_spectrogram(self.samples[index])
        label = self.labels[index]
        return torch.tensor(mel), torch.tensor(label)

    


