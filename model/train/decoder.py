import torch.nn as nn
import torch
import torch.nn.functional as F


MAX_LENGTH = 100

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers = 3, dropout_p=0.2, max_length=MAX_LENGTH, device ='cuda'):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers= n_layers, batch_first= True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        if device =='cuda':
            self.embedding = self.embedding.cuda()
            self.attn = self.attn.cuda()
            self.attn_combine = self.attn_combine.cuda()
            self.dropout = self.dropout.cuda()
            self.gru = self.gru.cuda()
            self.out = self.out.cuda()

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self,device):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)