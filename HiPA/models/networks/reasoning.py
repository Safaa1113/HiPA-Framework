from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MuHiPA.datasets.block as block
from torch.nn.utils.weight_norm import weight_norm
from HiPA.datasets.block.models.networks.vqa_net import mask_softmax
import math
from bootstrap.lib.logger import Logger
import time 




class HiPAReasoning(nn.Module):

    def __init__(self,
            residual=False,
            fusion_module_v = {},
            fusion_module_v2 = {},
            fusion_module_q = {},
            q_attention = False,
          ):
        super(HiPAReasoning, self).__init__()
        self.residual = residual
        self.fusion_module_v = fusion_module_v
        self.fusion_module_v2 = fusion_module_v2
        self.fusion_module_q = fusion_module_q

        if self.fusion_module_v:
            self.fusion_module_v = block.factory_fusion(self.fusion_module_v)
        if self.fusion_module_v2:
            self.fusion_module_v2 = block.factory_fusion(self.fusion_module_v2)
            
                
        if self.fusion_module_q:
            self.fusion_module_q = block.factory_fusion(self.fusion_module_q)
            
        self.q_attention = q_attention

        Logger().log_value('nparams_vfusion',
            self.get_nparams_vfusion(),
            should_print=True)
        Logger().log_value('nparams_qfusion',
            self.get_nparams_qfusion(),
            should_print=True)
        
        Logger().log_value('nparams of big cell',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)
        
        
    def get_nparams_vfusion(self):
        params = []
        if self.fusion_module_v:
            params += [p.numel() for p in self.fusion_module_v.parameters() if p.requires_grad]
            params += [p.numel() for p in self.fusion_module_v2.parameters() if p.requires_grad]
        return sum(params)
    
    def get_nparams_qfusion(self):
        params = []
        if self.fusion_module_q:
            params += [p.numel() for p in self.fusion_module_q.parameters() if p.requires_grad]
        return sum(params)    

    def forward(self, qq, mm):
        

        mm_new = mm
        qq_new = qq



  
        

        if self.fusion_module_v:

            mm_new = self.fusion_module_v2([mm_new, mm_new])
            mm_new = self.fusion_module_v([qq_new, mm_new])



        if self.fusion_module_q:
            qq_new = self.fusion_module_q([qq_new, mm_new])

            
        if self.q_attention:
  
             qq_new = qq * qq_new


        
        if self.residual:
            mm_new = mm_new + mm 
            qq_new = qq_new + qq 


        return mm_new, qq_new
