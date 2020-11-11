import torch
import random
from torch import optim
import torch.nn as nn
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from feature import load_audio, get_feature
import numpy as np
from utils import *


SOS_token = 0
EOS_token = 1


PATH = '../../model/'

#한 스텝 처리
def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, device='cuda'):
    encoder_hidden = encoder.initHidden()
    optimizer.zero_grad()

    target_length = len(target_tensor)

    encoder_outputs, encoder_hidden = encoder(input_tensor,encoder_hidden)

    loss = 0

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, torch.tensor([target_tensor[di]],device=device))
        decoder_input = torch.tensor([target_tensor[di]],device=device)

    loss.backward()

    optimizer.step()

    return loss / target_length


def trainEpoch(encoder, decoder, n_iters=1000, print_every=100, learning_rate=1e-2, resume=True,device='cuda'):
    start = time.time()
    writer = SummaryWriter('../../summary')

    print_loss_total = 0

    optimizer = optim.Adam([
        {'params':encoder.parameters()},
        {'params':decoder.parameters(),'lr':learning_rate/10}
    ],lr = learning_rate)

    iter = 0

    if resume == True:
        checkpoint = torch.load(PATH + 'model.tar',map_location=device)
        encoder.load_state_dict(checkpoint['encoder']).to(device)
        decoder.load_state_dict(checkpoint['decoder']).to(device)
        iter = checkpoint['iter']
        optimizer.load_state_dict(checkpoint['optimizer']).to(device)
        loss = checkpoint['loss']
        print_loss_total = checkpoint['pr_loss']

    train_set = get_train()
    char2id, id2char = load_label('../train_labels.csv')
    criterion = nn.NLLLoss().to(device)

    for it in range(n_iters):

        wave = load_audio(train_set[iter]+'.wav')
        data = get_feature(wave)
        input_tensor = torch.tensor(np.expand_dims(data,axis=1))

        with open(train_set[iter]+'.txt','r',encoding='utf-8') as file:
            txt = file.read()
        target_tensor = list(map(int,sentence_to_target(txt,char2id).split()))

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, device)

        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('[INFO] %s (%d %d%%) %.4f' % (timeSince(start, it / n_iters),
                                         it, it / n_iters * 100, print_loss_avg))
        iter += 1

        writer.add_scalar('Loss', loss, iter)
        writer.add_scalar()

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': iter,
        'loss': loss,
        'pr_loss': print_loss_total,
    }, PATH + 'model.tar')

