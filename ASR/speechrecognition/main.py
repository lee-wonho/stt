from model.train.train import *
from model.train.decoder import *
from model.train.encoder import *
from model.train.eval import *
from utils import *
import os

PATH = '../model/'

char2id , id2char = load_label()
print('[INFO] Label Loading Success!!')

encoder = VoiceEncoder(device='cuda')
decoder = AttnDecoderRNN(hidden_size=256, output_size=len(char2id))
optimizer = optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters(),'lr':1e-3}
    ],lr = 1e-2)

if os.path.exists(PATH+'model.tar'):
    checkpoint = torch.load(PATH + 'model.tar', map_location='cuda')
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to(encoder.device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(decoder.device)
    iter = checkpoint['iter']
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    print_loss_total = checkpoint['pr_loss']
print('[INFO] Model is loaded!!')


# summary wirter 정의 후 trainEpoch에 parameter로 전달 
# -> 학습 중간 오류로 인한 강제종료시 checkpoint와 summary간의 괴리를 줄이기 위해서

while True:
    done = trainEpoch(encoder, decoder, char2id, id2char, optimizer)

    if done:
        print('[INFO] The Train is over!!')
        break
