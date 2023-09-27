from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options, OptionsDict
from bootstrap.lib.logger import Logger
import MuHiPA.datasets.block as block
from HiPA.datasets.block.models.networks.vqa_net import factory_text_enc
from HiPA.datasets.block.models.networks.vqa_net import mask_softmax
from HiPA.datasets.block.models.networks.mlp import MLP
from .reasoning import HiPAReasoning
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import time 
import json


class HiPA(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            self_v_att={},
            n_step=3,
            shared=False,
            cell={},
            agg={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        super(MuHiPA, self).__init__()
        self.self_q_att = self_q_att
        self.self_q_att_2 = self_q_att_2
        self.n_step = n_step
        self.shared = shared
        self.self_v_att = self_v_att
        self.cell = cell
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean', 'none', 'sum']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        self.tag_embedding = torch.nn.Embedding(1601, 1240)
        
        
        
        self.mlp_glimpses_general = 2
        if self.self_q_att:
            self.mlp_glimpses_q = self.mlp_glimpses_general
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, self.mlp_glimpses_q)

        if self.self_v_att:
            self.mlp_glimpses = 2
            self.fusion = block.factory_fusion(self_v_att['fusion'])
            self.linear0 = nn.Linear(self_v_att['output_dim'], 512)
            self.linear1 = nn.Linear(512, self.mlp_glimpses_general)
            

        if self.shared:
            self.cell = MuHiPAReasoning(**cell)
        else:
            self.cells = nn.ModuleList([MuHiPAReasoning(**cell) for i in range(self.n_step)])
        

        if 'fusion' in self.classif:
            self.classif_module = block.factory_fusion(self.classif['fusion'])
        elif 'mlp' in self.classif:
            self.classif_module = MLP(self.classif['mlp'])
        else:
            raise ValueError(self.classif.keys())

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)
        
        Logger().log_value('nparams_q_attention',
            self.get_nparams_qattention(),
            should_print=True)
        
        Logger().log_value('nparams_v_attention',
            self.get_nparams_vattention(),
            should_print=True)
        
        Logger().log_value('nparams_class',
            self.get_nparams_class(),
            should_print=True)
        
        Logger().log_value('nparams_classifyer',
            self.get_nparams_classifyer(),
            should_print=True)

        self.buffer = None



    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        return sum(params)


    def get_nparams_qattention(self):
        params = []
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def get_nparams_vattention(self):
        
        params = []
        if self.self_v_att:
            params += [p.numel() for p in self.linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.linear1.parameters() if p.requires_grad]
            params += [p.numel() for p in self.fusion.parameters() if p.requires_grad]
        return sum(params)    

    def get_nparams_class(self):
        params = []

        params += [p.numel() for p in self.tag_embedding.parameters() if p.requires_grad]
        return sum(params)
    
    def get_nparams_classifyer(self):
        params = []

        params += [p.numel() for p in self.classif_module.parameters() if p.requires_grad]
        return sum(params)
    


    def set_pairs_ids(self, n_regions, bsize, device='cuda'):
        if self.shared and self.cell.pairwise:
            self.cell.pairwise_module.set_pairs_ids(n_regions, bsize, device=device)
        else:
            for i in self.n_step:
                if self.cells[i].pairwise:
                    self.cells[i].pairwise_module.set_pairs_ids(n_regions, bsize, device=device)


    def forward(self, batch):

        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        
        cls_score = batch['cls_scores']
        cls_text = batch['cls_text']
        cls_id = batch['cls']

        batch_size = v.shape[0]

        
        
        #tag processing
        cls_emb = self.tag_embedding(cls_id)
        cls_score = cls_score[:,:,None].expand(cls_score.size(0), cls_score.size(1), cls_emb.size(2))
        cls_emb *= cls_score
        t = cls_emb
        

        #process question
        q = self.process_question(q, l) #(20,4800)

        if self.self_q_att:
            q = self.question_attention(q, l)
            

            
        

        if self.self_v_att:
            v = torch.cat((t, v), 2)
            v = self.image_attention(q,v)


        qq = q 
        mm = v

        

        if self.cell:
            for i in range(self.n_step):
                
                cell = self.cell if self.shared else self.cells[i]
                mm, qq = cell(qq, mm)
                
            
       
        if self.agg['type'] == 'max':
            mm, mm_argmax = torch.max(mm, 1)
            qq, qq_argmax = torch.max(qq, 1)

        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)
            qq = qq.mean(1)


        elif self.agg['type'] == 'sum':
            mm = mm.sum(1)
            qq = qq.sum(1)

        elif self.agg['type'] == 'none':
            pass
            


        if 'fusion' in self.classif:

            logits = self.classif_module([qq, mm])
            
                
        elif 'mlp' in self.classif:
            logits = self.classif_module(mm)

       
        out = {'logits': logits}
        
        out = self.process_answers(out)

        return out
    

    
    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q_rnn, q_hidden = self.txt_enc.rnn(q_emb) #(bs, 10, 2400)

        return q_rnn
    
    def process_classif(self, q):
        q_classif = q[:,0:4]
        q_emb = self.txt_enc.embedding(q_classif)
        q_classif = q_emb.contiguous().view(q_emb.shape[0], -1)
        q_classif = self.q_classif_linear(q_classif)
        return q_classif
    
    def process_cls(self,cls_text):
        ids = []
        for i in range (len(cls_text)):
            text = [torch.LongTensor([self.word_to_wid.get(cls_text[i][j][k], 0) for k in range (len(cls_text[i][j]))])  for j in range (len(cls_text[i])) ]
            text = pad_sequence(text, padding_value=0, batch_first=True)
            text = torch.transpose(text, 0, 1)
            ids.append(text)

        input_cls = pad_sequence(ids, padding_value=0, batch_first=True)
        input_cls = torch.transpose(input_cls, 1, 2).cuda()

        cls_emb = self.txt_enc.embedding(input_cls)
        return input_cls, cls_emb

    def process_words(self, q):
        batch_size = q.shape[0]
        questions = []
        self.wid_to_word[0] = 'null'
        for i in range(batch_size):
            out= [self.wid_to_word[word_id.item()] for word_id in q[i]]
            questions.append(out)
        return questions

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()

        if range(batch_size) == range(0, 1):
            out['answers'] = [self.aid_to_ans[pred.item()] for i in range(batch_size)]
        else:
            out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
            
        if range(batch_size) == range(0, 1):
            out['answer_ids'] = [pred.item() for i in range(batch_size)]
        else:   
            out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out
    
    
    def question_attention(self, q, l):

        if self.self_q_att:
            q_att = self.q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = self.q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            buffer_whole_q = []
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)

                    q_out = q_att*q
                    buffer_whole_q.append(q_out)
  
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
                buffer_whole_q = torch.cat(buffer_whole_q, dim=2)

                buffer_argmax = buffer_whole_q.max(1)[1]

            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)

        return q
    

    
    def image_attention(self, q, v, mask=False):

        
        batch_size = q.size(0)
        n_regions = v.size(1)
        q = q[:,None,:].expand(q.size(0), n_regions, q.size(1))
        alpha = self.fusion([
            q.contiguous().view(batch_size*n_regions, -1),
            v.contiguous().view(batch_size*n_regions, -1)
        ])
        
        alpha = alpha.view(batch_size, n_regions, -1)


        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)

        alpha = F.softmax(alpha, dim=1)

        buffer_whole_v = []
        if alpha.size(2) > 1: # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                
                alpha = alpha.unsqueeze(2).expand_as(v)

                
                v_out = alpha*v
                buffer_whole_v.append(v_out)

                v_out = v_out.sum(1)
                v_outs.append(v_out)
            buffer_whole_v = torch.cat(buffer_whole_v, dim=2)
            buffer_argmax = buffer_whole_v.max(1)[1]
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha*v
            v_out = v_out.sum(1)
        return v_out  
    

  
