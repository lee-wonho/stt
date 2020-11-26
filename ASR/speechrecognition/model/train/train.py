import torch
from torch import optim
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from feature import load_audio, get_feature
import numpy as np
from utils import *


SOS_token = 0
EOS_token = 1

PATH = '../model/'

#한 스텝 처리
def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, char2id, id2char, device='cuda'):
    encoder_hidden = encoder.initHidden()
    optimizer.zero_grad()

    target_length = len(target_tensor)

    encoder_outputs, encoder_hidden = encoder(input_tensor,encoder_hidden)

    loss = 0

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    #decoder_outputs = []
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, torch.tensor([target_tensor[di]],device=device))
        decoder_input = torch.tensor([target_tensor[di]],device=device)

        # decoder_outputs.append(id2char[decoder_output.data.topk(1)[1].item()])

    loss.backward()

    optimizer.step()

    return loss / target_length #, decoder_outputs


def trainEpoch(encoder, decoder, char2id, id2char, optimizer, n_iters=1000, save_every=500, print_every=100,  resume=True,device='cuda'):
    BASE_PATH = '../wav/KsponSpeech/'
    start = time.time()
    # writer = SummaryWriter('../summary')
    done = False
    print_loss_total = 0

    iter = 0

    # if resume == True:
    #     checkpoint = torch.load(PATH + 'model.tar',map_location=device)
    #     encoder.load_state_dict(checkpoint['encoder'])
    #     encoder.to(device)
    #     decoder.load_state_dict(checkpoint['decoder'])
    #     decoder.to(device)
    #     iter = checkpoint['iter']
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     loss = checkpoint['loss']
    #     print_loss_total = checkpoint['pr_loss']

    train_set = get_train()
    criterion = nn.NLLLoss().to(device)

    for it in range(n_iters):
        if iter >= len(train_set):
            done = True
            break;
        wave = load_audio(BASE_PATH+train_set[iter]+'.wav')
        data = get_feature(wave)
        if device =='cpu':
            input_tensor = torch.tensor(np.expand_dims(data,axis=1))
        elif device == 'cuda':
            input_tensor = torch.cuda.FloatTensor(np.expand_dims(data,axis=1))
        with open(BASE_PATH+train_set[iter]+'.txt','r',encoding='utf-8') as file:
            txt = file.read()
        target_tensor = list(map(int,sentence_to_target(txt,char2id).split()))

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion,char2id, id2char, device)

        print_loss_total += loss

        if it % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('[INFO] %s (%d %d%%) %.4f' % (timeSince(start, it / n_iters),
                                         it, it / n_iters * 100, print_loss_avg))
            #print(''.join(output))
        iter += 1
        #
        # writer.add_scalar('Loss', loss, iter)


        # if it % save_every == 0 and it !=0:
        #     torch.save({
        #         'encoder': encoder.state_dict(),
        #         'decoder': decoder.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'iter': iter,
        #         'loss': loss,
        #         'pr_loss': print_loss_total,
        #     }, PATH + 'model.tar')
        #     print('[INFO] model saved!!')

        #if it != n_iters-1:
            #del data, wave, txt, target_tensor, loss, output, input_tensor

    print_loss_avg = print_loss_total / print_every
    print_loss_total = 0
    print('[INFO] %s (%d %d%%) %.4f' % (timeSince(start, it / n_iters),
                                        it, it / n_iters * 100, print_loss_avg))
    # print(''.join(output))

    # torch.save({
    #     'encoder': encoder.state_dict(),
    #     'decoder': decoder.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'iter': iter,
    #     'loss': loss,
    #     'pr_loss': print_loss_total,
    # }, PATH + 'model.tar')
    # print('[INFO] model saved!!')
    # return done