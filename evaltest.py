from utils import *
from model.train.encoder import VoiceEncoder
from model.train.decoder import AttnDecoderRNN
from feature import load_audio, get_feature

import numpy as np
import torch


PATH = '../model/'

SOS_token = 0
EOS_token = 1
max_length = 100

char2id, id2char = load_label()

encoder = VoiceEncoder(device='cuda')
decoder = AttnDecoderRNN(hidden_size = 256, output_size = len(char2id))

checkpoint = torch.load(PATH + 'model.tar', map_location='cuda')
encoder.load_state_dict(checkpoint['encoder'])
encoder.to(encoder.device)
decoder.load_state_dict(checkpoint['decoder'])
decoder.to(decoder.device)

path = '../wav/KsponSpeech/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000003'

wave = load_audio(path+'.wav')
data = get_feature(wave)

input_tensor = torch.cuda.FloatTensor(np.expand_dims(data,axis=1),device= 'cuda')

with open(path+'.txt','r',encoding='utf-8') as file:
    txt = file.read()

target_tensor = list(map(int, sentence_to_target(txt,char2id).split()))
encoder_hidden = encoder.initHidden()

encoder_outputs , encoder_hidden = encoder(input_tensor,encoder_hidden)

decoder_input = torch.tensor([[SOS_token]],device ='cuda')
decoder_hidden = encoder_hidden

decoded_words = []
decoder_attentions = torch.zeros(max_length,max_length)

for di in range(max_length):
    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs)
    decoder_attentions[di] = decoder_attention.data
    topv, topi = decoder_output.data.topk(1)

    if topi == EOS_token:
        decoded_words.append('<EOS>')
        break
    else:
        decoded_words.append(id2char[topi.item()])

    decoder_input = topi.squeeze().detach()

print(''.join(decoded_words))
print(txt)