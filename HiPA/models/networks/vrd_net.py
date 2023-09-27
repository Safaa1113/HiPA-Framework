import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import HiPA.datasets.block as block


class VRDNet(nn.Module):

    def __init__(self, opt):
        super(VRDNet, self).__init__()
        self.opt = opt
        self.classeme_embedding = nn.Embedding(
            self.opt['nb_classeme'],
            self.opt['classeme_dim'])
        self.fusion_c = block.factory_fusion(self.opt['classeme'])
        self.fusion_s = block.factory_fusion(self.opt['spatial'])
        self.fusion_f = block.factory_fusion(self.opt['feature'])

        self.predictor = MLP(**self.opt['predictor'])
       
        self.merge_c = block.fusions.Block(input_dims=[200,200], output_dim=200, mm_dim=300, rank=3, chunks=5)
        self.merge_f = block.fusions.Block(input_dims=[200,200], output_dim=200, mm_dim=300, rank=3, chunks=5)
        self.merge_s = block.fusions.Block(input_dims=[200,200], output_dim=200, mm_dim=300, rank=3, chunks=5)
        

  

    def forward(self, batch):
        
        bsize = batch['subject_boxes'].size(0)
        x_c = [self.classeme_embedding(batch['subject_cls_id']),
               self.classeme_embedding(batch['object_cls_id'])]
        x_s = [batch['subject_boxes'], batch['object_boxes']]
        x_f = [batch['subject_features'], batch['object_features']]

        x_c = self.fusion_c(x_c)
        x_s = self.fusion_s(x_s)
        x_f = self.fusion_f(x_f)
        
        for i in range(1):
            x_c_ = self.merge_c([x_c,x_c])
            x_s_ = self.merge_s([x_s,x_s])
            x_f_ = self.merge_f([x_f,x_f])
            x_c = x_c + x_c_
            x_s = x_s + x_s_
            x_f = x_f + x_f_
        

        
    
        x = torch.cat([x_c, x_s, x_f], -1)
        if 'aggreg_dropout' in self.opt:
            x = F.dropout(x, self.opt['aggreg_dropout'], training=self.training)
        y = self.predictor(x)
        

        
        out = {
            'rel_scores': y
        }
        
        return out


class MLP(nn.Module):
    
    def __init__(self,
            input_dim,
            dimensions,
            activation='relu',
            dropout=0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))
    
    def forward(self, x):
        for i,lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears)-1):
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x
