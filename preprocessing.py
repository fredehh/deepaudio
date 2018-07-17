import multiprocessing as mp
import os
import random
from functools import partial

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def build_spectrogram(file, output_dir="resources/data/"):
    y, sr = librosa.load(file[1] + file[0])
    n_fft = int(sr * 0.020)
    hop_len = int(sr * 0.010)
    spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.util.normalize(spectrogram, norm=np.inf)
    plot_spectrogram(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=2)
    np.save(output_dir + file[0].replace(".wav", "_stft"), spectrogram)


def build_mel_spectrogram(file, output_dir="resources/data/"):
    y, sr = librosa.load(file[1] + file[0])
    n_fft = int(sr * 0.020)
    n_mels = 128
    hop_len = int(sr * 0.010)
    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_len,
        power=2.0,
        n_mels=n_mels
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = librosa.util.normalize(spectrogram, norm=np.inf)
    plot_spectrogram(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=2)
    np.save(output_dir + file[0].replace(".wav", "_mel"), spectrogram)


def plot_spectrogram(spectrogram):
    librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def multiprocess(input_dir, func):
    files = [(f, input_dir) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    cpu_count = mp.cpu_count()
    pool = mp.Pool(cpu_count)
    pool.map(partial(func), files)
    pool.close()  # shut down the pool


def make_data(input, validation_ratio=.1, test_ratio=.2):
    dataset = pd.read_csv(input, names=["fold", "song1", "song2", "winner"])
    songs = dataset.iloc[:, 1:4]
    non_seen_songs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_songs1 = songs.loc[songs["song1"].isin(non_seen_songs)]
    test_songs2 = songs.loc[songs["song2"].isin(non_seen_songs)]
    test_songs = pd.concat([test_songs1, test_songs2])
    songs = songs.drop(test_songs.index)
    test_songs = test_songs.values
    test_songs = [flip(data) for data in test_songs]
    songs = songs.values
    songs = [flip(data) for data in songs]
    songs = shuffle(songs)
    validation_size = int(validation_ratio * len(songs))
    test_size = int(test_ratio * len(songs))
    if validation_ratio < test_ratio:
        validation, test, train = np.split(songs,
                                           [validation_size, test_size + validation_size])
    else:
        test, validation, train = np.split(songs,
                                           [test_size, validation_size + test_size])
    test_return = np.append(test_songs, test, axis=0)
    if os.path.exists("train.npy"):
        os.remove("test.npy")
        os.remove("validation.npy")
        os.remove("train.npy")
    np.save("test.npy", test_return)
    np.save("validation.npy", validation)
    np.save("train.npy", train)


def flip(data):
    if random.randint(0, 1) == 0:
        temp = data[0]
        data[0] = data[1]
        data[1] = temp
        data[2] = -1
    return data


# multiprocess("../resources/audio_wav/", build_spectrogram)
# multiprocess("../resources/audio_wav/", build_mel_spectrogram)
# make_data("../resources/annotations.csv")

#build_spectrogram(("0002.wav", "../resources/audio_wav/"))
#build_mel_spectrogram(("0002.wav", "../resources/audio_wav/"))
