import torch.nn as nn
import torch


class VoiceEncoder(nn.Module):
    def __init__(self, hidden_size=256, n_layers=3):
        super(VoiceEncoder,self).__init__()
        self.n_layers = n_layers
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0,20,inplace= True),
            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.MaxPool2d(2,stride=2),
        )

        self.birnn = nn.GRU(4096, hidden_size, num_layers=n_layers, batch_first=True, bidirectiomal=True)

    def forward(self, inputs, hidden):
        output = self.model(inputs)
        output, hidden = self.birnn(output,hidden)
        return output, hidden


    def initHidden(self,device):
        return torch.zeros(2*self.n_layers, 1, self.hidden_size, device=device)
