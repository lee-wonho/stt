from model.train.encoder import  VoiceEncoder
from model.train.decoder import AttnDecoderRNN
from model.train import train
from utils import load_label, sentence_to_target
from torch import optim

SOS_token = 0
EOS_token = 1

char2id , id2char = load_label()

encoder = VoiceEncoder(device='cuda')
decoder = AttnDecoderRNN(hidden_size = 256, output_size = len(char2id))
optimizer = optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters(),'lr':1e-3}
    ],lr = 1e-2)


#train.trainEpoch(encoder,decoder,resume=False)
train.trainEpoch(encoder,decoder, char2id=char2id,id2char=id2char,optimizer=optimizer)
train.trainEpoch(encoder,decoder, char2id=char2id,id2char=id2char,optimizer=optimizer)
train.trainEpoch(encoder,decoder, char2id=char2id,id2char=id2char,optimizer=optimizer)
train.trainEpoch(encoder,decoder, char2id=char2id,id2char=id2char,optimizer=optimizer)
train.trainEpoch(encoder,decoder, char2id=char2id,id2char=id2char,optimizer=optimizer)

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
