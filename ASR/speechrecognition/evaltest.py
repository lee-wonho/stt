from utils import *
from model.train.encoder import VoiceEncoder
from model.train.decoder import AttnDecoderRNN
from feature import load_audio, get_feature
from model.train import train


from torch import optim
import numpy as np
import torch
import wave, pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORED_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT, channels = CHANNELS, rate = RATE, input= True,
                frames_per_buffer = CHUNK)
print("recording")
frames= []
for i in range(0,int(RATE/CHUNK * RECORED_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Recorded")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


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

# path = '../wav/KsponSpeech/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000003'

path = WAVE_OUTPUT_FILENAME[:-4]
wave = load_audio(path+'.wav')
data = get_feature(wave)

input_tensor = torch.cuda.FloatTensor(np.expand_dims(data,axis=1),device= 'cuda')

# with open(path+'.txt','r',encoding='utf-8') as file:
#     txt = file.read()


# target_tensor = list(map(int, sentence_to_target(txt,char2id).split()))
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

# print(checkpoint['iter'])
print('Extracted : '+''.join(decoded_words))
# print('Raw : ' + txt)
