import os
import csv
import copy
import json
import torch
import numpy as np
from os import path as osp
from bootstrap.lib.logger import Logger
from .vqa_utils import AbstractVQA
from .vqa_utils import AbstractVizwiz
import re

class VQA2(AbstractVQA):

    def __init__(self,
            dir_data='data/vqa2',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn'):
        super(VQA2, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            has_valset=True,
            has_testset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False)
        self.dir_data2 = dir_data
        self.dir_rcnn = dir_rcnn
        # to activate manually in visualization context (notebook)
        self.load_original_annotation = False
        self.class_map = self.load_obj_vocab()

    
    def load_obj_vocab(self):
        
        # path = '/home/abr/Data/EMuRelPAFramework/EMuRelPA/datasets/block/datasets/'
        

        # p1 ='/EMuRelPA/datasets/block/datasets/objects_vocab.txt'
        # from pathlib import Path
        # sssss = str(Path().absolute()) + p1  
        sssss = self.dir_data2 + "/objects_vocab.txt"
        # f = open(os.path.join(path, "objects_vocab.txt"), "r")
        f = open(sssss, "r")
        f = f.read()
        f = f.split('\n')
        f = [re.split(" |,", f[i]) for i in range(len(f))]
        # re.split(" ,")
        f.insert(0,[""])
        f.pop()
        # print(f)
        # f = {}
        class_map = {i: l for i, l in enumerate(f)}
        # print(type(f))
        # print(class_map)
        return class_map

    def add_rcnn_to_item(self, item):
        path_rcnn = os.path.join(self.dir_rcnn, '{}.pth'.format(item['image_name']))
        item_rcnn = torch.load(path_rcnn)
        # print("hhhhhhhhhhhiiiiiiiiiiiiiiiiiiiii")
        # # print(type(item_rcnn))
        # for key, value in item_rcnn.items() :
        #     print (key)
        item['visual'] = item_rcnn['pooled_feat']
        item['coord'] = item_rcnn['rois']
        item['norm_coord'] = item_rcnn['norm_rois']
        item['nb_regions'] = item['visual'].size(0)
        item['cls_scores'] = item_rcnn['cls_scores']
        item['cls'] = item_rcnn['cls']
        item['cls_text'] = [self.class_map[int(item_rcnn['cls'][i])] for i in range(len(item_rcnn['cls']))]
        
        return item

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        if self.load_original_annotation:
            item['original_question'] = question

        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['image_name'] = question['image_name']
        
        # print("hhhhhhhhhhhiiiiiiiiiiiiiiiiiiiii")
        # for key, value in question.items() :
        #     print (key)

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            if self.load_original_annotation:
                item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                proba = proba / np.sum(proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']
        else:
            if item['question_id'] in self.is_qid_testdev:
                item['is_testdev'] = True
            else:
                item['is_testdev'] = False
        return item

    def download(self):
        dir_zip = osp.join(self.dir_raw, 'zip')
        os.system('mkdir -p '+dir_zip)
        dir_ann = osp.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_ann)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+dir_zip)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Test_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_train2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_train2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_val2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_val2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_train2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_val2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json'))






class Vizwiz(AbstractVizwiz):

    def __init__(self,
            dir_data='data/vqa2',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            annotations_in_top_answers_only = False,
            minwcount=10,
            nlp='mcb',
            att= True,
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn'):
        super(Vizwiz, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            att=att,
            annotations_in_top_answers_only = annotations_in_top_answers_only,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            has_valset=True,
            has_testset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False)
        self.dir_rcnn = dir_rcnn
        # to activate manually in visualization context (notebook)
        self.load_original_annotation = False
        # self.class_map = self.load_obj_vocab()

    
    def load_obj_vocab(self):
        
        path = '/home/abr/Data/EMuRelPAFramework/EMuRelPA/datasets/block/datasets/'
        # path = os.path.relpath(path, start = os.curdir)
        f = open(os.path.join(path, "objects_vocab.txt"), "r")
        f = f.read()
        f = f.split('\n')
        f = [re.split(" |,", f[i]) for i in range(len(f))]
        # re.split(" ,")
        f.insert(0,[""])
        f.pop()
        # print(f)
        # f = {}
        class_map = {i: l for i, l in enumerate(f)}
        # print(type(f))
        # print(class_map)
        return class_map

    def add_rcnn_to_item(self, item):
        
        feature_id = item['image_id']
        if (item['mode']=="train"):
            item['visual'] = self.visual_train[feature_id]
            if self.att:
                # print(item['visual'].shape)
                item['visual'] = item['visual'].view(item['visual'].shape[1],item['visual'].shape[2]
                                                     ,item['visual'].shape[0])
        elif  (item['mode']=="val"):
            item['visual'] = self.visual_val[feature_id]
            if self.att:
                item['visual'] = item['visual'].view(item['visual'].shape[1],item['visual'].shape[2]
                                                     ,item['visual'].shape[0])

            # features = self.hdf5_file_val['att']
            # item['visual_att'] = torch.from_numpy(features[feature_id].astype('float32'))
            # features = self.hdf5_file_val['noatt']
            # item['visual_no_att'] = torch.from_numpy(features[feature_id].astype('float32'))
            # item['visual_att'] = self.hdf5_file_val['att'][feature_id]
            # item['visual_no_att'] = self.hdf5_file_val['noatt'][feature_id]
            # item['visual_image_name'] = self.hdf5_file_val['img_name'][feature_id]
        elif  (item['mode']=="test"):
            item['visual'] = self.visual_test[feature_id]
            if self.att:
                item['visual'] = item['visual'].view(item['visual'].shape[1],item['visual'].shape[2]
                                                     ,item['visual'].shape[0])
            # features = self.hdf5_file_test['att']
            # item['visual_att'] = torch.from_numpy(features[feature_id].astype('float32'))
            # features = self.hdf5_file_test['noatt']
            # item['visual_no_att'] = torch.from_numpy(features[feature_id].astype('float32'))
            # item['visual_att'] = self.hdf5_file_test['att'][feature_id]
            # item['visual_no_att'] = self.hdf5_file_test['noatt'][feature_id]
            # item['visual_image_name'] = self.hdf5_file_test['img_name'][feature_id]

        
        return item

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        if self.load_original_annotation:
            item['original_question'] = question

        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        # print(question.keys())
        item['image_name'] = question['image']
        item['image_id'] = question['image_id']
        item['mode'] = question['mode']

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'answer' in self.dataset['questions'][0].keys():
            annotation = self.dataset['questions'][index]
            # if self.load_original_annotation:
            #     item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                # print("-----------------------",proba)
                proba = proba / np.sum(proba)
                # print("[[[[[[[[[[[[[[[[[", proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['answer_type']
        # else:
        #     if item['question_id'] in self.is_qid_testdev:
        #         item['is_testdev'] = True
        #     else:
        #         item['is_testdev'] = False
        return item

    def download(self):
        dir_zip = osp.join(self.dir_raw, 'zip')
        os.system('mkdir -p '+dir_zip)
        dir_ann = osp.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_ann)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+dir_zip)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Test_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_train2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_train2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_val2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_val2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_train2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_val2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json'))

