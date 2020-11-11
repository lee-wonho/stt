from model.train.encoder import  VoiceEncoder
from model.train.decoder import AttnDecoderRNN
from feature import get_feature, load_audio
import numpy as np
from torch import Tensor, LongTensor
import torch.nn as nn
import pandas as pd


SOS_token = 0
EOS_token = 1

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



path = '../wav/KsponSpeech/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001'

wave = load_audio(path+'.wav')
with open(path+'.txt','r',encoding='utf-8') as file:
    txt = file.read()

char2id , id2char = load_label('../train_labels.csv')
embedded = sentence_to_target(txt,char2id)
target = list(map(int,embedded.split()))

encoder = VoiceEncoder(device='cpu')
decoder = AttnDecoderRNN(hidden_size = 256, output_size = len(char2id))

data = get_feature(wave)
ten = Tensor(np.expand_dims(data,axis=1))
hidden = encoder.model(ten)

embedding = nn.Embedding(2040,512)

init = encoder.initHidden()

encoder_outputs , encoder_hidden = encoder(ten,init)

print(ten.shape)
print(hidden.shape)
print(encoder_outputs.shape)
print(encoder_hidden.shape)

target_length = len(target)
decoder_input = LongTensor([SOS_token])

decoder_words = []
decoder_hidden = encoder_hidden

print(embedding(decoder_input).shape)
print(decoder_input)
for di in range(target_length):
    decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

    topv, topi = decoder_output.topk(1)
    topi = np.asarray(topi)[0,0]
    decoder_words.append(id2char[topi])

    decoder_input = LongTensor([target[di]])
    print(decoder_input)
    print(decoder_words)

    if decoder_input == EOS_token:
        break
