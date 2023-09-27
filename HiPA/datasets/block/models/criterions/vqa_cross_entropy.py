import torch.nn as nn
import torch
import json
from pathlib import Path
class VQACrossEntropyLoss(nn.Module):

    def __init__(self):
        super(VQACrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        
        self.loss2 = nn.CosineSimilarity()
        # p2 = '/averaged_dict00.json'
        # p1 ='/EMuRelPA/models/networks/averaged_dict00.json'
        
        # with open( str(Path().absolute()) + p1 ) as json_file:
        #     data = json.load(json_file)
        # self.data = data
        # count 0 none of the above 1 number 2 color 3 time 4 yes/no 5 there 6 name of something 7 who 8 what name 9 what is the woman 10
        self.q_type_dic = {"what is" : 12, "is this":5, "how many": 0,
                      "what time" : 4, "are there any" : 6,
                      "what is this" : 9, "what is the": 11,
                      "what color is the":3, "is there a": 6,
                      "none of the above": 1, "what kind of": 7,
                      "what" : 13, "what is in the": 9,
                      "are they": 15, "what type of":7,
                      "is this a":5, "how many people are": 0,
                      "where are the": 17, "what does the": 14,
                      "do you": 18, "is this person": 17, "was":16,
                      "why is the": 19, "what color is": 3,
                      "how": 20, "what room is": 9, "do" :21,
                      "is the": 5, "how many people are in":0,
                      "are there": 6, "are these": 5, "is he": 17,
                      "what color are the": 3, "what is the name":7,
                      "can you" : 22, "which":23, "what animal is": 7,
                      "what is on the" : 9, "is the person": 17,
                      "what sport is" :7, 
                      "why" : 24, "is this a" : 5, "is the woman" : 17,
                      "is that a": 5, "are":16, "what color" : 3,
                      "is it" : 25, "what is the person": 10,
                      "is there" : 6,  "where is the" : 8,
                      "what" : 26, "is there a": 6, "does this": 5,
                      "what is the man":10, "what are the" : 11,
                      "what number is" : 2,
                      "is":16, "what are":12, "is the man":5,
                      "is this an": 5, "are the" : 5, "what is the woman":10,
                      "does the" : 5, "what brand" :7,
                      "who is" : 8, "what is the color of the": 3, "has": 28,
                      "could":29
     
                }    
        # self.q_type_dic = {"what is" : 0, "is this":1, "how many": 2,
        #               "what time" : 3, "are there any" : 4,
        #              "what is this" : 5, "what is the": 6,
        #              "what color is the":7, "is there a": 8,
        #              "none of the above": 9, "what kind of": 10,
        #              "what" : 11, "what is in the": 12,
        #              "are they": 13, "what type of":14,
        #              "is this a":15, "how many people are": 16,
        #              "where are the": 17, "what does the": 18,
        #              "do you": 19, "is this person": 20, "was":21,
        #              "why is the": 22, "what color is": 23,
        #              "how": 24, "what room is": 25, "do" :26,
        #              "is the": 27, "are they":28, "how many people are in":29,
        #              "are there": 30, "are these": 31, "is he": 32,
        #              "what color are the": 33, "what is the name":34,
        #              "can you" : 35, "which":36, "what animal is": 37,
        #              "what is on the" : 38, "is the person": 39,
        #              "what sport is" : 40, "what is in the":41,
        #              "why" : 42, "is this a" : 43, "is the woman" : 44,
        #              "is that a": 45, "are":46, "what color" : 47,
        #              "is it" : 48, "what is the person": 49,
        #              "is there" : 50,  "where is the" : 8,
        #              "what" : 51, "is there a": 52, "does this": 53,
        #              "what is the man":54, "what are the" : 55,
        #              "what is the" : 56, "what number is" : 57,
        #              "is":58, "what are":59, "is the man":60, "what":61,
        #              "is this an": 62, "are the" : 63, "what is the woman":64,
        #              "does the" : 65, "what brand" : 66, "what is this":67,
        #              "who is" : 68, "what is the color of the": 69, "has": 70,
        #              "could":71
     
        #         }        
 
        
        

    def forward(self, net_out, batch):
        
        
        # print("File      Path:", Path(__file__).absolute())
        # print("Directory Path:", Path().absolute()) # Directory of current working directory, not __file__  

        
        
        
        # data[an1][1]['agg_Q']
        # data[an1][0]['agg_I']
        # data[an1][1]['cellFinal_Q_RF']
        # data[an1][0]['cellFinal_I_RF'] 
        
        
        # net_out['v_agg']
        # net_out['q_agg']
        # net_out['v_reas']
        # net_out['q_reas']
        
        # embdloss = 0
        
        
        # for j in range(len(batch['answer'])): 
        #     embdloss2 = 0
        #     k = batch['answer'][j]
        #     if k in self.data.keys():
        #         # agg_Q = torch.FloatTensor(self.data[k][1]['agg_Q']).cuda()
        #         # embdloss2 += 1 - self.loss2(net_out['q_agg'][j].unsqueeze(0), agg_Q.unsqueeze(0))
                
        #         # agg_V = torch.FloatTensor(self.data[k][0]['agg_I']).cuda()
        #         # embdloss2 += 1 - self.loss2(net_out['v_agg'][j].unsqueeze(0), agg_V.unsqueeze(0))
                
        #         # reas_Q = torch.FloatTensor(self.data[k][1]['cellFinal_Q_RF']).cuda()
        #         # embdloss2 += 1 - self.loss2(net_out['q_reas'][j].unsqueeze(0), reas_Q.unsqueeze(0))
                
        #         reas_V = torch.FloatTensor(self.data[k][0]['cellFinal_I_RF'] ).cuda()
        #         embdloss2 += 1 - self.loss2(net_out['v_reas'][j].unsqueeze(0), reas_V.unsqueeze(0))
                
        #         if not torch.isnan(embdloss2):
        #             embdloss += embdloss2
        # # print (embdloss)
        # embdloss = embdloss * (0.0001)
        
        #qt = batch['question_type']
        # q_type_id = [self.q_type_dic[qtx] for qtx in batch['question_type']]
        # q_type_id = torch.tensor(q_type_id).cuda()
        #print("fffffffffffffffffffffffffffffffffffff")
        #print(batch['class_id'].shape)
        #print(q_type_id.shape)
        out = {}
        # print("-----------", net_out['q_classif'].shape)
        # print("-----------", q_type_id)
        # print("-----------", net_out['logits'].shape)
        # print("-----------", batch['class_id'])
        
        # out['loss'] = embdloss + self.loss(
        #     net_out['logits'],
        #     batch['class_id'].squeeze(1)) 
        
        
        out['loss'] =  self.loss(
            net_out['logits'],
            batch['class_id'].squeeze(1)) 
        
        # print("*.*"*20)
        # print(out)
        # print(net_out)
        # print(batch['class_id'])
        # print("*.*"*20)
        # print(l)
        # + self.loss(
        # net_out['q_classif'],
        #     q_type_id)
        # out['loss'] = self.loss(
        #     net_out['logits'],
        #     batch['class_id'].squeeze(1))
        return out







class VQACrossEntropyLossEDC(nn.Module):

    def __init__(self):
        super(VQACrossEntropyLossEDC, self).__init__()
        self.loss1 = nn.CrossEntropyLoss()
        
        self.loss_2 = nn.CrossEntropyLoss()
        
        self.loss_3 = nn.CrossEntropyLoss()
 
    def forward(self, net_out, batch):
        
        # print(net_out['logits'].shape)
        # print(net_out['logits_2'].shape)
        # print(net_out['logits_3'].shape)
        # print(batch['class_id'].shape)
        # print(batch['class_id'].squeeze(1).shape)
        # print(batch['class_id'].shape)
        # print(h)

        out = {}

        loss1 =  self.loss1(
            net_out['logits'],
            batch['class_id'].squeeze(1)) 
        
        loss2 = self.loss_2(
            net_out['logits_2'],
            batch['class_id'].squeeze(1)) *0.01 #* 0.01  
        
        loss3 = self.loss_3(
            net_out['logits_3'],
            batch['class_id'].squeeze(1)) *0.001 #* 0.001 
        
        out['loss'] = loss1+loss2+loss3
        
        return out






class VRDBCELoss(nn.Module):

    def __init__(self):
        super(VRDBCELoss, self).__init__()
        self.loss0 = nn.BCEWithLogitsLoss()
        self.loss1 = nn.BCEWithLogitsLoss()

    def forward(self, net_output, target):
        y_true = target['target_oh']
        # print("-"*30)
        # print(type(net_output['rel_scores']))
        # print(u)
        cost0 = self.loss0(net_output['rel_scores'][0], y_true) * .001 #*0 #*.00001
        cost1 = self.loss1(net_output['rel_scores'][1], y_true)
        cost = cost0 + cost1
        out = {}
        out['loss'] = cost
        return out