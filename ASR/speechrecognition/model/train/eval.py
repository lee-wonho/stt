import torch
from utils import *
from feature import get_feature
import numpy as np



MAX_LENGTH = 50
SOS_token = 0
EOS_token = 1

def evaluate(encoder, decoder, wave, max_length=MAX_LENGTH, device ='cuda'):
    with torch.no_grad():
        char2id , id2char = load_label()

        data = get_feature(wave)

        input_tensor = torch.tensor(np.expand_dims(data,axis=1),device=device) # test 데이터 셋에 대한 호출

        encoder_hidden = encoder.initHidden()

        encoder_outputs, encoder_hidden = encoder(input_tensor,encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(id2char[topi])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]