import torch.nn as nn
import models.modules as modules

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()
        self.E = Encoder #(2048,8,10)
        self.DCE = modules.DCE(num_features,num_features//2, sizes=(1, 2, 3, 6))
        self.BUBF = modules.BUBF(num_features,64)
        self.D = modules.Decoder(num_features)
        self.SRM = modules.SRM(num_features)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_dce = self.DCE(x_block4)
        x_bubf = self.BUBF(x_block1, x_block2, x_block3, x_block4)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4,x_dce)
        out = self.SRM(x_decoder,x_bubf)
        return out
