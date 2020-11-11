from model.train.encoder import  VoiceEncoder
from feature import get_feature, load_audio
import numpy as np
from torch import Tensor, LongTensor
import torch.nn as nn
import pandas as pd


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding='cp949')
    id_list = ch_labels['id']
    char_list = ch_labels['char']

    for id, char in zip(id_list, char_list):
        char2id[char] = id
        id2char[id] = char

    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = ""
    for ch in sentence:
        target += (str(char2id[ch]) + ' ')
    return target[:-1]


encoder = VoiceEncoder(device='cpu')

path = '../wav/KsponSpeech/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001'

wave = load_audio(path+'.wav')
with open(path+'.txt','r',encoding='utf-8') as file:
    txt = file.read()

char2id , id2char = load_label('../train_labels.csv')
embedded = sentence_to_target(txt,char2id)
target = list(map(int,embedded.split()))


data = get_feature(wave)
ten = Tensor(np.expand_dims(data,axis=1))
hidden = encoder.model(ten)

embedding = nn.Embedding(2040,512)

vec = embedding(LongTensor(target))


init = encoder.initHidden()

print(encoder(ten,init).shape)