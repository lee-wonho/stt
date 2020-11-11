import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from torch import(
Tensor,
FloatTensor
)
import os
import matplotlib.pyplot as plt
import librosa
import random
from argparse import ArgumentParser


def load_audio(audio_path, del_silence=False):
    signal = np.memmap(audio_path, dtype='h',mode='r').astype('float32')

    if del_silence:
        non_silence_indices = split(signal, top_db=30)
        signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

    return signal / 32767

#Split an audio signal into non-silent intervals.
def split(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    return librosa.effects.split(y, top_db, ref, frame_length, hop_length)


class MelSpectrogram(object):
    def __init__(self, sample_rate=16000, n_mels=80, frame_length = 20, frame_shift=10):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = int(round(sample_rate*0.001*frame_length))
        self.hop_length=int(round(sample_rate*0.001*frame_shift))
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, win_length=frame_length,
            hop_length=self.hop_length, n_fft=self.n_fft,
            n_mels=n_mels,
            window_fn=torch.hann_window
        )
    def __call__(self, signal):
        melspectrogram = self.transforms(Tensor(signal))
        melspectrogram = self.amplitude_to_db(melspectrogram)
        melspectrogram = melspectrogram.numpy()

        return melspectrogram

class MFCC(object):
    def __init__(self, sample_rate=16000, n_mfcc=40, dct_type = 2):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type

        self.transforms = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            dct_type = dct_type, norm = 'ortho',
            log_mels = False
        )

    def __call__(self, signal):
        mfcc = self.transforms(FloatTensor(signal))
        mfcc = mfcc.numpy()

        return mfcc


class FilterBank(object):
    def __init__(self, sample_rate=16000, n_mels=80, frame_length=20, frame_shift=10):
        self.transforms = torchaudio.compliance.kaldi.fbank
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return self.transforms(
            Tensor(signal).unsqueeze(0), num_mel_bins=self.n_mels,
            frame_length=self.frame_length, frame_shift=self.frame_shift,
            sample_frequency = self.sample_rate,
            use_log_fbank= True,
            window_type='hanning'
        ).transpose(0,1).numpy()

class SpecAugment(object):
    def __init__(self, freq_mask_para=18, time_mask_num=10, freq_mask_num=2)->None:
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num

    def __call__(self, feature_vector:Tensor)->Tensor:
        feature_vector = Tensor(feature_vector)
        time_axis_length = feature_vector.size(0)
        freq_axis_length = feature_vector.size(1)
        time_mask_para = time_axis_length/20

        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature_vector[t0:t0+t, :]=0

        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature_vector[:, f0: f0+f]=0

        return feature_vector

#써도 되고 안써도 되고...
class NoiseInjector(object):
    def __init__(self, dataset_path, noiseset_size, sample_rate=16000, noise_level=0.7):
        print("Create Noise injector")

        self.noiseset_size = noiseset_size
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.audio_paths = self.create_audio_paths(dataset_path)
        self.dataset = self.create_noiseset(dataset_path)

        print("Create Noise injector complete")

    def __call__(self, signal):
        noise = np.random.choice(self.dataset)
        noise_level = np.random.uniform(0, self.noise_level)

        signal_length = len(signal)
        noise_length = len(noise)

        if signal_length >= noise_length:
            noise_start = int(np.random.rand() * (signal_length - noise_length))
            noise_end = int(noise_start + noise_length)
            signal[noise_start : noise_end] += noise * noise_level

        else:
            signal += noise[:signal_length]*noise_level

        return signal

    def create_audio_paths(self, dataset_path):
        audio_paths = list()
        data_list = os.listdir(dataset_path)
        data_list_size = len(data_list)

        while True:
            index = int(random.random() * data_list_size)

            if data_list[index].endswith('.wav'):
                audio_paths.append(data_list[index])

            if len(audio_paths)==self.noiseset_size:
                break

        return audio_paths

    def create_noiseset(self, dataset_path):
        dataset = list()

        for audio_path in self.audio_paths:
            path = os.path.join(dataset_path, audio_path)
            noise = self.extract_noise(path)

            if noise is not None:
                dataset.append(noise)

        return dataset

    def extract_noise(self, audio_path):
        try:
            signal = np.memmap(audio_path, dtype='h', mode = 'r').astype('float32')
            non_silence_indices = split(signal, top_db=30)

            for (start, end) in non_silence_indices:
                signal[start:end]=0

            noise=signal[signal!=0]
            return noise / 32767

        except RuntimeError:
            print("RuntimeError in {0}".format(audio_path))
            return None

        except ValueError:
            print("ValueError in {0}".format(audio_path))
            return None
        
#kospeech에 기본적으로 존재하는 parser.py import만 바꿔주면 될듯


if __name__ =='__main__':
    wave  = load_audio('../KsponSpeech_000001.wav')

    fb = FilterBank()



def get_feature(wave, method='fb'):
    if method =='fb':
        f = FilterBank()
    elif method == 'mfcc':
        f = MFCC()
    elif method =='mel':
        f = MelSpectrogram()
    else:
        raise ValueError('Method {} is not proper '.format(method))

    features = []
    data = f(wave)
    size = 0

    while True:
        if size + 20 > np.shape(data)[1]:
            a = 20 - (np.shape(data)[1] - size)
            features.append(np.append(data[:, size:],np.zeros((80, a)), axis =1 ))
            break
        else:
            features.append(data[:, size:size+20])

        size += 20

    return np.stack(features,axis=0)