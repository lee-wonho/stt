import torch.nn as nn
import torch
from torch import Tensor

class VoiceEncoder(nn.Module):
    def __init__(self,max_length=100, hidden_size=256, n_layers=3, device = 'cuda'):
        super(VoiceEncoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=2, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0,20,inplace= True),
            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,stride=2),
        )

        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=n_layers)

        if device == 'cuda':
            self.model = self.model.cuda('cuda')
            self.rnn = self.rnn.cuda('cuda')

    def forward(self, inputs, hidden):
        input = self.model(inputs)

        outputs = torch.zeros(self.max_length, self.hidden_size, device=self.device)

        input = input.reshape((input.shape[0],25,1,self.hidden_size))
        input_length = input.size()[0]

        for ei in range(input_length):
            output, hidden = self.rnn(input[ei],hidden)
            outputs[ei] += output[0, 0]
            if ei+1 >= self.max_length:
                break
                # 여기서 문제 터지는데 outputs의 길이가 100이 max라서 문제가 생기는 듯?  outputs = (100,256) output = (100,256)

        return outputs, hidden


    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device)
