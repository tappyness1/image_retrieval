import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_end_dim(hw):
    """helper function so that the dimension of the img (after convolving) will be exactly what can be fed into FCN

    Args:
        hw (_type_): size to be defined eg 64x64 -> 64

    Returns:
        _type_: integer of the end result
    """
    conv_res_1  = (hw - 5 + 1)
    maxpool_res_1 = (conv_res_1 - 2) / 2 + 1
    conv_res_2 = maxpool_res_1 - 5 + 1 
    end_res = (conv_res_2 - 2) / 2 + 1
    return int(end_res)

class SiameseNetwork(nn.Module):
    def __init__(self, emb_dim=128, hw = 64):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.end_dim = calc_end_dim(hw)

        self.fc = nn.Sequential(
            nn.Linear(64*self.end_dim*self.end_dim, 512), # 64x64 => 13x13, 224x224 => 53x54
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        # print (x.size())
        x = x.view(-1, 64*self.end_dim*self.end_dim) # 64x64 => 13x13, 224x224 => 53x54
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x