import torch
import torch.nn as nn
from objectives import cca_loss

#方法一：

class Networks(nn.Module):
    def __init__(self):
        super(Networks, self).__init__()
        self.encoder1 = nn.Sequential(
             nn.Linear(1750, 620, bias=False),
             nn.ReLU(),
             #nn.Linear(500, 300, bias=False),
             #nn.ReLU(),
             #nn.Linear(300, 100, bias=False),
            #nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
            #nn.Linear(100, 300, bias=False),
            #nn.ReLU(),
            #nn.Linear(300, 500, bias=False),
            #nn.ReLU(),
            nn.Linear(620, 1750, bias=False),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(79, 620, bias=False),
            nn.ReLU(),
            #nn.Linear(500, 300, bias=False),
            #nn.ReLU(),
            #nn.Linear(300, 100, bias=False),
            #nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            #nn.Linear(100, 300, bias=False),
            #nn.ReLU(),
            #nn.Linear(300, 500, bias=False),
            #nn.ReLU(),
            nn.Linear(620, 79, bias=False),
            nn.ReLU(),
        )

        self.model1 = nn.Linear(620, 10)
        self.model2 = nn.Linear(620, 10)
        self.loss = cca_loss(5).loss
        self.weight = nn.Parameter(1.0e-4 * torch.ones(2500, 2500))
        self.beta1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.beta2 = torch.nn.Parameter(torch.Tensor([0.5]))
        
    def forward(self, input1, input2):
        
        output1 = self.encoder1(input1)
        output1 = self.decoder1(output1)

        output2 = self.encoder2(input2)
        output2 = self.decoder2(output2)
        
        return output1, output2
        
    def forward2(self, input1, input2):
        coef = self.weight - torch.diag(torch.diag(self.weight))

        out_1 = self.encoder1(input1)
        out_11 = out_1[0:2250, :]
        out11_ = self.model1(out_11)
        out_12 = out_1[2250:2375, :]
        
        out_2 = self.encoder2(input2)
        out_21 = out_2[0:2250, :]
        out12_ = self.model1(out_21)
        out_22 = out_2[2250:2375, :]

        out_com = (self.beta1 * out_11 + self.beta2 * out_21) / (self.beta1 + self.beta2)
        out_com = torch.cat((out_com, out_12), 0)
        out_com = torch.cat((out_com, out_22), 0)
        
        output_coef = torch.matmul(coef, out_com)
        
        out11 = output_coef[0:2375, :]
        output11 = self.decoder1(out11)

        out22 = torch.cat((output_coef[0:2250, :], output_coef[2375:2500, :]), 0)
        output22 = self.decoder2(out22)
                
        return output11, output22, out_1, out_2, out11_, out12_, coef, out11, out22