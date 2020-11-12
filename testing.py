from model.train.encoder import  VoiceEncoder
from model.train.decoder import AttnDecoderRNN
from feature import get_feature, load_audio
import numpy as np
from torch import Tensor
import torch.nn as nn
import pandas as pd
from torch import optim
import torch
from model.train import train
from utils import load_label, sentence_to_target


SOS_token = 0
EOS_token = 1

path = '../wav/KsponSpeech/'\
       'KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000004'

wave = load_audio(path+'.wav')
with open(path+'.txt','r',encoding='utf-8') as file:
    txt = file.read()

char2id , id2char = load_label()
embedded = sentence_to_target(txt,char2id)
target = list(map(int,embedded.split()))


# optimizer = optim.Adam([
#     {'params': encoder.parameters()},
#     {'params': decoder.parameters(), 'lr': 1e-3}
# ], lr=1e-2)
# criterion = nn.NLLLoss().cuda()
#
# data = get_feature(wave)
# ten = Tensor(np.expand_dims(data,axis=1))
# ten = ten.cuda()
# hidden = encoder.model(ten)
#
# embedding = nn.Embedding(2040,512)
#
# init = encoder.initHidden()
# loss = 0
# encoder_outputs , encoder_hidden = encoder(ten,init)
#
# print(ten.shape)
# print(hidden.shape)
# print(encoder_outputs.shape)
# print(encoder_hidden.shape)
#
# target_length = len(target)
# decoder_input = torch.tensor([[SOS_token]])
#
# decoder_words = []
# decoder_hidden = encoder_hidden
#
# print(embedding(decoder_input).shape)
# print(decoder_input)
#
# optimizer.step()

# for di in range(target_length):
#     decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
#
#     topv, topi = decoder_output.topk(1)
#     topi = np.asarray(topi)[0,0]
#     decoder_words.append(id2char[topi])
#
#     decoder_input = LongTensor([target[di]])
#     print(decoder_words)
#     loss += criterion(decoder_output, LongTensor([target[di]]))
#
#     if decoder_input == EOS_token:
#         break
# loss.backward()
# encoder_optim.step()
# decoder_optim.step()

# loss = train.train(ten,target,encoder,decoder,optimizer,criterion,device='cuda')

encoder = VoiceEncoder(device='cuda')
decoder = AttnDecoderRNN(hidden_size = 256, output_size = len(char2id))

train.trainEpoch(encoder,decoder,resume=False)

