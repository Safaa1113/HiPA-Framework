import os
import re
import sys
import json
import torch
import torch.utils.data as data
import numpy as np
from os import path as osp
from tqdm import tqdm
from collections import Counter
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.datasets.dataset import Dataset
from bootstrap.datasets import transforms as bootstrap_tf
from bootstrap.datasets.dataset import ListDatasets

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def tokenize_mcb(s):
    t_str = s.lower()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub( i, '', t_str)
    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
    q_list = re.sub(r'\?','',t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list


class AbstractVQA(Dataset):

    def __init__(self,
            dir_data='/local/cadene/data/vqa',
            split='train', 
            batch_size=80,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            has_valset=True,
            has_testset=True,
            has_testset_anno=False,
            has_testdevset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False):
        super(AbstractVQA, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        
        
        
        self.nans = nans
        self.minwcount = minwcount
        self.nlp = nlp
        self.proc_split = proc_split
        self.samplingans = samplingans
        # preprocessing
        self.has_valset = has_valset
        self.has_testset = has_testset
        self.has_testset_anno = has_testset_anno
        self.has_testdevset = has_testdevset
        self.has_answers_occurence = has_answers_occurence
        self.do_tokenize_answers = do_tokenize_answers

        # sanity checks
        if self.split in ['test', 'val'] and self.samplingans:
            raise ValueError()

        self.dir_raw = os.path.join(self.dir_data, 'raw')
        if not os.path.exists(self.dir_raw):
            self.download()

        self.dir_processed = os.path.join(self.dir_data, 'processed')
        self.subdir_processed = self.get_subdir_processed()
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')
        self.path_trainset = osp.join(self.subdir_processed, 'trainset.pth')
        self.path_valset = osp.join(self.subdir_processed, 'valset.pth')
        self.path_is_qid_testdev = osp.join(self.subdir_processed, 'is_qid_testdev.pth')
        self.path_testset = osp.join(self.subdir_processed, 'testset.pth')
        
        if not os.path.exists(self.subdir_processed):
            self.process()

        self.wid_to_word = torch.load(self.path_wid_to_word)
        self.word_to_wid = torch.load(self.path_word_to_wid)
        self.aid_to_ans = torch.load(self.path_aid_to_ans)
        self.ans_to_aid = torch.load(self.path_ans_to_aid)

        if 'train' in self.split:
            self.dataset = torch.load(self.path_trainset)
            # print("="*20)
            # print(type(self.dataset['questions']))
            # self.dataset['questions'] = self.dataset['questions'][0:500]
        elif self.split == 'val':
            if self.proc_split == 'train':
                self.dataset = torch.load(self.path_valset)
                # self.dataset['questions'] = self.dataset['questions'][0:500]
            elif self.proc_split == 'trainval':
                self.dataset = torch.load(self.path_trainset)
                # self.dataset['questions'] = self.dataset['questions'][0:500]
        elif self.split == 'test':
            self.dataset = torch.load(self.path_testset)
            # self.dataset['questions'] = self.dataset['questions'][0:500]
            if self.has_testdevset:
                self.is_qid_testdev = torch.load(self.path_is_qid_testdev)

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.PadTensors(use_keys=[
                'question', 'pooled_feat', 'cls_scores', 'rois', 'cls', 'cls_oh', 'norm_rois'
            ]),
            #bootstrap_tf.SortByKey(key='lengths'), # no need for the current implementation
            bootstrap_tf.StackTensors()
        ])

        if self.proc_split == 'trainval' and self.split in ['train','val']:
            self.bootstrapping()

    def add_word_tokens(self, word_to_wid):
        for word, wid in word_to_wid.items():
            if word not in self.word_to_wid:
                self.word_to_wid[word] = len(self.word_to_wid)+1 # begin at 1, last is UNK
        self.wid_to_word = {wid:word for word, wid in self.word_to_wid.items()}

    def bootstrapping(self):
        rnd = np.random.RandomState(seed=Options()['misc']['seed'])
        indices = rnd.choice(len(self),
                             size=int(len(self)*0.95),
                             replace=False)
        if self.split == 'val':
            indices = np.array(list(set(np.arange(len(self))) - set(indices)))
        self.dataset['questions'] = [self.dataset['questions'][i] for i in indices]
        self.dataset['annotations'] = [self.dataset['annotations'][i] for i in indices]

    def __len__(self):
        return len(self.dataset['questions'])

    def get_image_name(self, image_id='1', format='COCO_%s_%012d.jpg'):
        return format%(self.get_subtype(), image_id)

    def name_subdir_processed(self):
        subdir = 'nans,{}_minwcount,{}_nlp,{}_proc_split,{}'.format(
            self.nans, self.minwcount, self.nlp, self.proc_split)
        return subdir

    def get_subdir_processed(self):
        name = self.name_subdir_processed()
        subdir = os.path.join(self.dir_processed, name)
        return subdir

    def get_subtype(self, testdev=False):
        if testdev:
            return 'test-dev2015'
        if self.split in ['train', 'val']:
            return self.split+'2014'
        elif self.split == 'test':
            return self.split+'2015'
        elif self.split == 'testdev':
            return 'test-dev2015'

    ################################################################################
    # Preprocessing

    def download(self):
        raise NotImplementedError()

    def process(self):
        dir_ann = osp.join(self.dir_raw, 'annotations')
        path_train_ann = osp.join(dir_ann, 'mscoco_train2014_annotations.json')
        path_train_ques = osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json')
        path_val_ann = osp.join(dir_ann, 'mscoco_val2014_annotations.json')
        path_val_ques = osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json')
        path_test_ques = osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json')
        path_test_ann = osp.join(dir_ann, 'mscoco_test2015_annotations.json')
        path_testdev_ques = osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json')
        
        train_ann = json.load(open(path_train_ann))
        train_ques = json.load(open(path_train_ques))
        trainset = self.merge_annotations_with_questions(train_ann, train_ques)
        # trainset = train_ques
        
        if self.has_valset:
            val_ann = json.load(open(path_val_ann))
            val_ques = json.load(open(path_val_ques))
            valset = self.merge_annotations_with_questions(val_ann, val_ques)
            # valset = val_ques

        if self.has_testset:
            test_ques = json.load(open(path_test_ques))
        
            if self.has_testset_anno:
                test_ann = json.load(open(path_test_ann))
                testset = self.merge_annotations_with_questions(test_ann, test_ques)
                # testset = test_ann
            else:
                testset = test_ques

        if self.has_testdevset:
            testdev = json.load(open(path_testdev_ques))

        trainset = self.add_image_names(trainset)

        if self.has_valset:
            valset = self.add_image_names(valset)

        if self.has_testset:
            testset = self.add_image_names(testset)

        if self.has_testdevset:
            testdev = self.add_image_names(testdev)

        if self.proc_split == 'trainval' and self.has_valset:
            trainset['questions'] += valset['questions']
            trainset['annotations'] += valset['annotations']
            
            
        print("------------------------ add answer")    
            
        trainset['annotations'] = self.add_answer(trainset['annotations'])
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.add_answer(valset['annotations'])
        if self.has_testset_anno:
            testset['annotations'] = self.add_answer(testset['annotations'])

        if self.do_tokenize_answers:
            trainset['annotations'] = self.tokenize_answers(trainset['annotations'])
            if self.proc_split == 'train' and self.has_valset:
                valset['annotations'] = self.tokenize_answers(valset['annotations'])

        top_answers = self.top_answers(trainset['annotations'], self.nans)
        aid_to_ans = [a for i,a in enumerate(top_answers)]
        ans_to_aid = {a:i for i,a in enumerate(top_answers)}

        trainset['questions'] = self.tokenize_questions(trainset['questions'], self.nlp)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.tokenize_questions(valset['questions'], self.nlp)
        if self.has_testset:
            testset['questions'] = self.tokenize_questions(testset['questions'], self.nlp)

        # Creation of word dictionary made before removing questions
        # Bigger dictionary == Better chance to model words from testset == Better generalization
        top_words, wcounts = self.top_words(trainset['questions'], self.minwcount)
        wid_to_word = {i+1:w for i,w in enumerate(top_words)}
        word_to_wid = {w:i+1 for i,w in enumerate(top_words)}

        # Remove the questions not in top answers
        trainset['annotations'], trainset['questions'] = self.annotations_in_top_answers(
            trainset['annotations'], trainset['questions'], aid_to_ans)

        trainset['questions'] = self.insert_UNK_token(trainset['questions'], wcounts, self.minwcount)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.insert_UNK_token(valset['questions'], wcounts, self.minwcount)
        if self.has_testset:
            testset['questions'] = self.insert_UNK_token(testset['questions'], wcounts, self.minwcount)
        
        trainset['questions'] = self.encode_questions(trainset['questions'], word_to_wid)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.encode_questions(valset['questions'], word_to_wid)
        if self.has_testset:
            testset['questions'] = self.encode_questions(testset['questions'], word_to_wid)

        trainset['annotations'] = self.encode_answers(trainset['annotations'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.encode_answers(valset['annotations'], ans_to_aid)
        if self.has_testset_anno:
            testset['annotations'] = self.encode_answers(testset['annotations'], ans_to_aid)

        if self.has_answers_occurence:
            trainset['annotations'] = self.add_answers_occurence(trainset['annotations'], ans_to_aid)

        Logger()('Save processed datasets to {}'.format(self.subdir_processed))
        os.system('mkdir -p '+self.subdir_processed)
        if self.has_testdevset:
            is_qid_testdev = {item['question_id']:True for item in testdev['questions']}
            torch.save(is_qid_testdev, self.path_is_qid_testdev)
        torch.save(wid_to_word, self.path_wid_to_word)
        torch.save(word_to_wid, self.path_word_to_wid)
        torch.save(aid_to_ans, self.path_aid_to_ans)
        torch.save(ans_to_aid, self.path_ans_to_aid)
        torch.save(trainset, self.path_trainset)
        if self.proc_split == 'train' and self.has_valset:
            torch.save(valset, self.path_valset)
        if self.has_testset:
            torch.save(testset, self.path_testset)

    def tokenize_answers(self, annotations):
        Logger()('Example of modified answers after preprocessing:')
        for i, ex in enumerate(annotations):
            s = ex['answer']
            if self.nlp == 'nltk':
                ex['answer'] = " ".join(word_tokenize(str(s).lower()))
            elif self.nlp == 'mcb':
                ex['answer'] = " ".join(tokenize_mcb(s))
            elif self.nlp == 'bert':
                ex['answer'] = " ".join(tokenize_mcb(s))
            else:
                ex['answer'] = " ".join(tokenize(s))
            if i < 10: Logger()('{} became -> {} <-'.format(s,ex['answer']))
            if i>0 and i % 1000 == 0:
                sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(annotations), i*100.0/len(annotations)) )
                sys.stdout.flush() 
        return annotations

    def add_image_names(self, dataset):
        for q in dataset['questions']:
            q['image_name'] = 'COCO_%s_%012d.jpg'%(dataset['data_subtype'],q['image_id'])
        return dataset

    def add_answer(self, annotations):
        for item in annotations:
            # print(item.keys())
            # print(item['question_type'])
            # print(item['multiple_choice_answer'])
            # print(item['answers'])
            # print(item['image_id'])
            # print(item['answer_type'])
            # print(item['question_id'])
            # print(item['question_id_'])
            # print("----------------------------------------------")
            item['answer'] = item['multiple_choice_answer']
        return annotations

    def top_answers(self, annotations, nans):
        counts = {}
        for item in tqdm(annotations):
            # print(item['answer'])
            # print(item['answer_'])
            ans = item['answer'] 
            counts[ans] = counts.get(ans, 0) + 1

        cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
        Logger()('Top answer and their counts:')
        for i in range(20):
            Logger()(cw[i])

        vocab = []
        for i in range(nans):
            vocab.append(cw[i][1])
        Logger()('Number of answers left: {} / {}'.format(len(vocab), len(cw)))
        print(vocab[:nans])
        return vocab[:nans]

    def annotations_in_top_answers(self, annotations, questions, top_answers):
        new_anno = []
        new_ques = []
        if len(annotations) != len(questions): # sanity check
            raise ValueError()
        for i in tqdm(range(len(annotations))):
            if annotations[i]['answer'] in top_answers:
                new_anno.append(annotations[i])
                new_ques.append(questions[i])
        Logger()('Number of examples reduced from {} to {}'.format(
            len(annotations), len(new_anno)))
        return new_anno, new_ques

    def tokenize_questions(self, questions, nlp):
        Logger()('Tokenize questions')
        if nlp == 'nltk':
            from nltk.tokenize import word_tokenize
        for item in tqdm(questions):
            ques = item['question']
            if nlp == 'nltk':
                item['question_tokens'] = word_tokenize(str(ques).lower())
            elif nlp == 'mcb':
                item['question_tokens'] = tokenize_mcb(ques)
            elif nlp == 'bert':
                # print("*/"*50)
                # print("Tokenizing With BERT "*10)
                from transformers import BertTokenizer, BertModel
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                marked_text = "[CLS] " + ques + " [SEP]"
                tokenized_text = tokenizer.tokenize(marked_text)
                item['question_tokens'] = tokenized_text
            else:
                item['question_tokens'] = tokenize(ques)
        return questions

    def top_words(self, questions, minwcount):
        wcounts = {}
        for item in questions:
            for w in item['question_tokens']:
                wcounts[w] = wcounts.get(w, 0) + 1
        cw = sorted([(count,w) for w, count in wcounts.items()], reverse=True)
        Logger()('Top words and their wcounts:')
        for i in range(20):
            Logger()(cw[i])

        total_words = sum(wcounts.values())
        Logger()('Total words: {}'.format(total_words))
        bad_words = [w for w in sorted(wcounts) if wcounts[w] <= minwcount]
        vocab = [w for w in sorted(wcounts) if wcounts[w] > minwcount]
        bad_count = sum([wcounts[w] for w in bad_words])
        Logger()('Number of bad words: {}/{} = {:.2f}'.format(len(bad_words), len(wcounts), len(bad_words)*100.0/len(wcounts)))
        Logger()('Number of words in vocab would be {}'.format(len(vocab)))
        Logger()('Number of UNKs: {}/{} = {:.2f}'.format(bad_count, total_words, bad_count*100.0/total_words))
        vocab.append('UNK')
        return vocab, wcounts

    def merge_annotations_with_questions(self, ann, ques):
        for key in ann:
            if key not in ques:
                ques[key] = ann[key]
                
        
        return ques

    def insert_UNK_token(self, questions, wcounts, minwcount):
        for item in questions:
            item['question_tokens_UNK'] = [w if wcounts.get(w,0) > minwcount else 'UNK' for w in item['question_tokens']]
        return questions

    def encode_questions(self, questions, word_to_wid):
        for item in questions:
            item['question_wids'] = [word_to_wid[w] for w in item['question_tokens_UNK']]
        return questions

    def encode_answers(self, annotations, ans_to_aid):
        for item in annotations:
            # associate to examples when answer is not in classes dictionnary
            # the class_id of the less occuring answer
            # Warning: the accuracy may be a bit wrong (to use only during validation)
            item['answer_id'] = ans_to_aid.get(item['answer'], len(ans_to_aid)-1)
        return annotations

    def add_answers_occurence(self, annotations, ans_to_aid):
        # for samplingans during training
        for item in annotations:
            item['answers_word'] = []
            item['answers_id'] = []
            item['answers_count'] = []
            answers = [a['answer'] for a in item['answers']]
            for ans, count in dict(Counter(answers)).items():
                if ans in ans_to_aid:
                    item['answers_word'].append(ans)
                    item['answers_id'].append(ans_to_aid[ans])
                    item['answers_count'].append(count)
        return annotations

    #############################################################################""
    # Preprocessing on a list dataset (vqa2+vgenome setup)

    def sync_from(self, dataset):
        Logger()('Sync {}.{} from {}.{}'.format(
            self.__class__.__name__,
            self.split,
            dataset.__class__.__name__,
            dataset.split))

        if 'annotations' in self.dataset:
            Logger()('Removing triplets with answer not in dict answer')
            list_anno = []
            list_ques = []
            for i in tqdm(range(len(self))):
                ques = self.dataset['questions'][i]
                answer = self.dataset['annotations'][i]['answer']
                if answer in dataset.ans_to_aid:
                    # sync answer_id
                    anno = self.dataset['annotations'][i]
                    anno['answer_id'] = dataset.ans_to_aid[answer]

                    list_anno.append(anno)
                    list_ques.append(ques)
            self.dataset['annotations'] = list_anno
            self.dataset['questions'] = list_ques
            Logger()('{} / {} remaining questions after filter_from'.format(len(list_ques), len(self.dataset['questions'])))

        # sync question_wids
        Logger()('Sync question_wids')
        for i in tqdm(range(len(self))):
            ques = self.dataset['questions'][i]
            for j, token in enumerate(ques['question_tokens']):
                if token in dataset.word_to_wid:
                    ques['question_tokens_UNK'][j] = token
                    ques['question_wids'][j] = dataset.word_to_wid[token]
                else:
                    ques['question_tokens_UNK'][j] = 'UNK'
                    ques['question_wids'][j] = dataset.word_to_wid['UNK']

        # sync dict word, dict ans
        self.word_to_wid = dataset.word_to_wid
        self.wid_to_word = dataset.wid_to_word
        self.ans_to_aid = dataset.ans_to_aid
        self.aid_to_ans = dataset.aid_to_ans





class ListVQADatasets(ListDatasets):

    def __init__(self,
             datasets,
             split='train',
             batch_size=4,
             shuffle=False,
             pin_memory=False,
             nb_threads=4,
             seed=1337):
        super(ListVQADatasets, self).__init__(
            datasets=datasets,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            bootstrapping=False,
            seed=seed)

        self.subdir_processed = self.make_subdir_processed()
        Logger()('Subdir proccessed: {}'.format(self.subdir_processed))
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')

        self.process()
        
        # if not os.path.isdir(self.subdir_processed):
        #     self.process()
        # else:
        #     Logger()('Loading list_datasets_vqa proccessed state')
        #     self.wid_to_word = torch.load(self.path_wid_to_word)
        #     self.word_to_wid = torch.load(self.path_word_to_wid)
        #     self.aid_to_ans = torch.load(self.path_aid_to_ans)
        #     self.ans_to_aid = torch.load(self.path_ans_to_aid)

        #     for i in range(len(self.datasets)):
        #         subdir_processed = os.path.join(self.subdir_processed, '{}.{}'.format(
        #             self.datasets[i].__class__.__name__,
        #             self.datasets[i].split))
        #         path_dataset = os.path.join(subdir_processed, 'dataset.pth')
        #         self.datasets[i].dataset = torch.load(path_dataset)
        #     Logger()('Done !')

        Logger()('Final number of tokens {}'.format(len(self.word_to_wid)))

        self.make_lengths_and_ids()
        

    def make_subdir_processed(self):
        processed = ''
        for i in range(0, len(self.datasets)):
            processed += '{}.{}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split,
                self.datasets[i].name_subdir_processed())
            if i < len(self.datasets)-1:
                processed += '+'
        self.subdir_processed = osp.join(self.datasets[0].dir_processed, processed)
        return self.subdir_processed

    def process(self):
        os.system('mkdir -p '+self.subdir_processed)

        for i in range(1,len(self.datasets)):
            Logger()('Add word tokens of {}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split))
            self.datasets[0].add_word_tokens(self.datasets[i].word_to_wid)

        for i in range(len(self.datasets)):
            Logger()('Sync {}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split))
            self.datasets[i].sync_from(self.datasets[0])

        self.word_to_wid = self.datasets[0].word_to_wid
        self.wid_to_word = self.datasets[0].wid_to_word
        self.ans_to_aid = self.datasets[0].ans_to_aid
        self.aid_to_ans = self.datasets[0].aid_to_ans

        Logger()('Saving list_datasets_vqa proccessed state')
        torch.save(self.wid_to_word, self.path_wid_to_word)
        torch.save(self.word_to_wid, self.path_word_to_wid)
        torch.save(self.aid_to_ans, self.path_aid_to_ans)
        torch.save(self.ans_to_aid, self.path_ans_to_aid)

        for i in range(len(self.datasets)):
            subdir_processed = os.path.join(self.subdir_processed, '{}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split))
            os.system('mkdir -p '+subdir_processed)

            path_dataset = os.path.join(subdir_processed, 'dataset.pth')
            torch.save(self.datasets[i].dataset, path_dataset)
        Logger()('Done !')

    def get_subtype(self):
        return self.datasets[0].get_subtype()


























#------------------------------------------vizwiz





import h5py
import torch
import torch.utils.data as data


class FeaturesDataset(data.Dataset):

    def __init__(self, features_path, mode):
        self.path_hdf5 = features_path

        assert os.path.isfile(self.path_hdf5), \
            'File not found in {}, you must extract the features first with images_preprocessing.py'.format(
                self.path_hdf5)

        self.hdf5_file = h5py.File(self.path_hdf5, 'r')
        self.dataset_features = self.hdf5_file[mode]  # noatt or att (attention)

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset_features[index].astype('float32'))

    def __len__(self):
        return self.dataset_features.shape[0]



class AbstractVizwiz(Dataset):

    def __init__(self,
            dir_data='data/vizwiz',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            annotations_in_top_answers_only = False,
            nlp='mcb',
            att=True,
            proc_split='train',
            samplingans=False,
            has_valset=True,
            has_testset=True,
            has_testset_anno=False,
            has_testdevset=False,
            has_answers_occurence=True,
            do_tokenize_answers=False):
        super(AbstractVizwiz, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        self.nans = nans
        self.minwcount = minwcount
        self.annotations_in_top_answers_only = annotations_in_top_answers_only
        self.nlp = nlp
        self.proc_split = proc_split
        self.samplingans = samplingans
        self.att = att
        # preprocessing
        self.has_valset = has_valset
        self.has_testset = has_testset
        self.has_testset_anno = has_testset_anno
        self.has_testdevset = has_testdevset
        self.has_answers_occurence = has_answers_occurence
        self.do_tokenize_answers = do_tokenize_answers

        # sanity checks
        if self.split in ['test', 'val'] and self.samplingans:
            raise ValueError()

        self.dir_raw = os.path.join(self.dir_data)
        if not os.path.exists(self.dir_raw):
            self.download()

        self.dir_processed = os.path.join(self.dir_data, 'processed')
        self.subdir_processed = self.get_subdir_processed()
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')
        self.path_trainset = osp.join(self.subdir_processed, 'trainset.pth')
        self.path_valset = osp.join(self.subdir_processed, 'valset.pth')
        self.path_is_qid_testdev = osp.join(self.subdir_processed, 'is_qid_testdev.pth')
        self.path_testset = osp.join(self.subdir_processed, 'testset.pth')
        
        if not os.path.exists(self.subdir_processed):
            self.process()

        self.wid_to_word = torch.load(self.path_wid_to_word)
        self.word_to_wid = torch.load(self.path_word_to_wid)
        self.aid_to_ans = torch.load(self.path_aid_to_ans)
        self.ans_to_aid = torch.load(self.path_ans_to_aid)
        

        # self.hdf5_file_train = h5py.File(self.dir_raw +'/resnet14x14.h5', 'r')
        # self.hdf5_file_val = h5py.File(self.dir_raw +'/resnet14x14v.h5', 'r')
        # self.hdf5_file_test = h5py.File(self.dir_raw +'/resnet14x14vt.h5', 'r')
        
        if self.att:
            mode = "att"
        else:
            mode = "noatt"

        self.visual_train = FeaturesDataset(self.dir_raw +'/resnet14x14.h5', mode)
        self.visual_val = FeaturesDataset(self.dir_raw +'/resnet14x14v.h5', mode)
        self.visual_test = FeaturesDataset(self.dir_raw +'/resnet14x14vt.h5', mode)

        if 'train' in self.split:
            self.dataset = torch.load(self.path_trainset)
        elif self.split == 'val':
            if self.proc_split == 'train':
                self.dataset = torch.load(self.path_valset)
            elif self.proc_split == 'trainval':
                self.dataset = torch.load(self.path_trainset)
        elif self.split == 'test':
            self.dataset = torch.load(self.path_testset)
            if self.has_testdevset:
                self.is_qid_testdev = torch.load(self.path_is_qid_testdev)

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.PadTensors(use_keys=[
                'question', 'pooled_feat', 'cls_scores', 'rois', 'cls', 'cls_oh', 'norm_rois'
            ]),
            #bootstrap_tf.SortByKey(key='lengths'), # no need for the current implementation
            bootstrap_tf.StackTensors()
        ])

        if self.proc_split == 'trainval' and self.split in ['train','val']:
            self.bootstrapping()

    def add_word_tokens(self, word_to_wid):
        for word, wid in word_to_wid.items():
            if word not in self.word_to_wid:
                self.word_to_wid[word] = len(self.word_to_wid)+1 # begin at 1, last is UNK
        self.wid_to_word = {wid:word for word, wid in self.word_to_wid.items()}

    def bootstrapping(self):
        rnd = np.random.RandomState(seed=Options()['misc']['seed'])
        indices = rnd.choice(len(self),
                             size=int(len(self)*0.95),
                             replace=False)
        if self.split == 'val':
            indices = np.array(list(set(np.arange(len(self))) - set(indices)))
        self.dataset['questions'] = [self.dataset['questions'][i] for i in indices]
        self.dataset['annotations'] = [self.dataset['annotations'][i] for i in indices]

    def __len__(self):
        return len(self.dataset['questions'])

    def get_image_name(self, image_id='1', format='COCO_%s_%012d.jpg'):
        return format%(self.get_subtype(), image_id)

    def name_subdir_processed(self):
        subdir = 'nans,{}_minwcount,{}_nlp,{}_proc_split,{}'.format(
            self.nans, self.minwcount, self.nlp, self.proc_split)
        return subdir

    def get_subdir_processed(self):
        name = self.name_subdir_processed()
        subdir = os.path.join(self.dir_processed, name)
        return subdir

    def get_subtype(self, testdev=False):
        if testdev:
            return 'test-dev2015'
        if self.split in ['train', 'val']:
            return self.split+'2014'
        elif self.split == 'test':
            return self.split+'2015'
        elif self.split == 'testdev':
            return 'test-dev2015'

    ################################################################################
    # Preprocessing

    def download(self):
        raise NotImplementedError()
        
    def prepare_questions(self, annotations):
        """ Filter, Normalize and Tokenize question. """
        

        prepared = []
        questions = [q['question'] for q in annotations]
        dataset['question']
        for key, value in annotations[0].items() :
            print (key)
    
        for question in questions:
            # lower case
            # question = question.lower()
    
            # # define desired replacements here
            # punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}
            # conversational_dict = {"thank you": '', "thanks": '', "thank": '', "please": '', "hello": '',
            #                        "hi ": ' ', "hey ": ' ', "good morning": '', "good afternoon": '', "have a nice day": '',
            #                        "okay": '', "goodbye": ''}
    
            # rep = punctuation_dict
            # rep.update(conversational_dict)
    
            # # use these three lines to do the replacement
            # rep = dict((re.escape(k), v) for k, v in rep.items())
            # pattern = re.compile("|".join(rep.keys()))
            # question = pattern.sub(lambda m: rep[re.escape(m.group(0))], question)
    
            # # sentence to list
            # question = question.split(' ')
    
            # # remove empty strings
            # question = list(filter(None, question))
    
            prepared.append(question)
    
        return prepared
    def prepare_answers(self, annotations):
        answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
        prepared = []
        
        for item in annotations:
            item['answer'] = item['multiple_choice_answer']
        for sample_answers in answers:
            prepared_sample_answers = []
            for answer in sample_answers:
                # lower case
                answer = answer.lower()
    
                # define desired replacements here
                punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}
    
                rep = punctuation_dict
                rep = dict((re.escape(k), v) for k, v in rep.items())
                pattern = re.compile("|".join(rep.keys()))
                answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
                prepared_sample_answers.append(answer)
    
            prepared.append(prepared_sample_answers)
        return prepared
    
    
    def encode_answers(self, answers, answer_to_index):
        answer_vec = torch.zeros(len(answer_to_index))
        for answer in answers:
            index = answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec
    
    def add_id_to_dataset(self, annotations , name_to_id, mode="train"):
        # print(name_to_id)
        for q in annotations['questions']:
            # print("---------------------------------")
            # print(q.keys())
            q['image_id'] = name_to_id[q['image']]
            q['mode'] = mode

        # print (self.annotations_train[0]['image'])
        
        return annotations
    
    
    def add_vfeatures_to_dataset(self, annotations , visual_features):
        print(annotations.keys())
        # print(name_to_id)
        for q in annotations:
            # q['image_id'] = name_to_id[q['image']]
            # print("---------------------------------")
            # print(q.keys())
            feature_id = q['image_id']
            q['visual_att'] = visual_features['att'][feature_id]
            q['visual_no_att'] = visual_features['noatt'][feature_id]
            # print(visual_features.keys())
            q['visual_image_name'] = visual_features['img_name'][feature_id]
        
        return annotations   
    
    def prepare_answers(self, annotations):
        answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
        prepared = []
    
        for sample_answers in answers:
            prepared_sample_answers = []
            for answer in sample_answers:
                # lower case
                answer = answer.lower()
    
                # define desired replacements here
                punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}
    
                rep = punctuation_dict
                rep = dict((re.escape(k), v) for k, v in rep.items())
                pattern = re.compile("|".join(rep.keys()))
                answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
                prepared_sample_answers.append(answer)
    
            prepared.append(prepared_sample_answers)
        return prepared

    def process(self):
        
        
        annotations_train = {}
        annotations_val = {}
        annotations_test = {}
        
        annotations_dir = osp.join(self.dir_raw, 'Annotations')
        
        path_ann_train = os.path.join(annotations_dir, 'train' + ".json")
        with open(path_ann_train, 'r') as fd:
            annotations_train['questions'] = json.load(fd)
            
        
        if self.has_valset:
            path_ann_val = os.path.join(annotations_dir, 'val' + ".json")
            with open(path_ann_val, 'r') as fd:
                annotations_val['questions'] = json.load(fd)

        if self.has_testset:
            path_ann_test = os.path.join(annotations_dir, 'test' + ".json")
            with open(path_ann_test, 'r') as fd:
                annotations_test['questions'] = json.load(fd)
        
        self.max_question_length = 26
        # self.split = split

        # vocab
        with open(osp.join(self.dir_raw, 'vocabs.json'), 'r') as fd:
            vocabs = json.load(fd)
        self.vocabs = vocabs
      
        
        import h5py
        with h5py.File(self.dir_raw +'/resnet14x14.h5', 'r') as f:
            img_names = f['img_name'][()]
        self.name_to_id_train = {name: i for i, name in enumerate(img_names)}
        
        annotations_train = self.add_id_to_dataset(annotations_train ,self.name_to_id_train, mode="train")
        
        if self.has_valset:
            with h5py.File(self.dir_raw +'/resnet14x14v.h5', 'r') as f:
                img_names = f['img_name'][()]
            self.name_to_id_val = {name: i for i, name in enumerate(img_names)}
            annotations_val = self.add_id_to_dataset(annotations_val ,self.name_to_id_val, mode="val")

        if self.has_testset:
            with h5py.File(self.dir_raw +'/resnet14x14vt.h5', 'r') as f:
                img_names = f['img_name'][()]
            self.name_to_id_test = {name: i for i, name in enumerate(img_names)}
            annotations_test = self.add_id_to_dataset(annotations_test ,self.name_to_id_test, mode="test")

        
        # load features
        self.hdf5_file_train = h5py.File(self.dir_raw +'/resnet14x14.h5', 'r')
        # print(self.hdf5_file.keys())
        # print(self.hdf5_file['img_name'])
        
        # self.annotations_train = self.add_vfeatures_to_dataset(self.annotations_train , self.hdf5_file_train)
        
        if self.has_valset:
            self.hdf5_file_val = h5py.File(self.dir_raw +'/resnet14x14v.h5', 'r')
            # self.annotations_val = self.add_vfeatures_to_dataset(self.annotations_val , self.hdf5_file_val)
            
        if self.has_testset:
            self.hdf5_file_test = h5py.File(self.dir_raw +'/resnet14x14vt.h5', 'r')
            # self.annotations_test = self.add_vfeatures_to_dataset(self.annotations_test , self.hdf5_file_test)
    
        
        prepared_answers = self.prepare_answers(annotations_train['questions'])
        import itertools
        prepared_answers = itertools.chain.from_iterable(prepared_answers)
        from collections import Counter
        counter = Counter(prepared_answers)
        counted_ans = counter.most_common(self.nans)
        # start from labels from 0
        self.prepared_answers_anw_to_id = {t[0]: i for i, t in enumerate(counted_ans, start=0)}
        # print(self.prepared_answers_anw_to_id)
    
        # vocab = {t[0]: i for i, t in enumerate(counted_ans, start=0)}
        
        aid_to_ans = [a[0] for i,a in enumerate(counted_ans)]
        ans_to_aid = {a[0]:i for i,a in enumerate(counted_ans)}
  
        if self.proc_split == 'trainval' and self.has_valset:
            annotations_train['questions'] += annotations_val['questions']
            # trainset['annotations'] += valset['annotations']

        annotations_train['questions'] = self.add_answer(annotations_train['questions'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            annotations_val['questions'] = self.add_answer(annotations_val['questions'], ans_to_aid)
        if self.has_testset_anno:
            annotations_test['questions'] = self.add_answer(annotations_test['questions'], ans_to_aid)

        if self.do_tokenize_answers:
            annotations_train['questions'] = self.tokenize_answers(annotations_train['questions'])
            if self.proc_split == 'train' and self.has_valset:
                annotations_val['questions'] = self.tokenize_answers(annotations_val['questions'])

        # top_answers = self.top_answers(annotations_train['questions'], self.nans)
        # aid_to_ans = [a for i,a in enumerate(top_answers)]
        # ans_to_aid = {a:i for i,a in enumerate(top_answers)}
        
        # aid_to_ans = [a[0] for i,a in enumerate(counted_ans)]
        # ans_to_aid = {a[0]:i for i,a in enumerate(counted_ans)}
        
        # self.prepared_answers_anw_to_id = {t: i for i, t in enumerate(counted_ans, start=0)}

        annotations_train['questions'] = self.tokenize_questions(annotations_train['questions'], self.nlp)
        if self.proc_split == 'train' and self.has_valset:
            annotations_val['questions'] = self.tokenize_questions(annotations_val['questions'], self.nlp)
        if self.has_testset:
            annotations_test['questions'] = self.tokenize_questions(annotations_test['questions'], self.nlp)

        # Creation of word dictionary made before removing questions
        # Bigger dictionary == Better chance to model words from testset == Better generalization
        top_words, wcounts = self.top_words(annotations_train['questions'], self.minwcount)
        wid_to_word = {i+1:w for i,w in enumerate(top_words)}
        word_to_wid = {w:i+1 for i,w in enumerate(top_words)}
        
        

        # Remove the questions not in top answers
        
        
        annotations_train['questions'] = self.annotations_in_top_answers(
                annotations_train['questions'], aid_to_ans)
        
        
        annotations_train['questions'] = self.insert_UNK_token(annotations_train['questions'], wcounts, self.minwcount)
        
        if self.proc_split == 'train' and self.has_valset:
            annotations_val['questions'] = self.insert_UNK_token(annotations_val['questions'], wcounts, self.minwcount)
        if self.has_testset:
            annotations_test['questions'] = self.insert_UNK_token(annotations_test['questions'], wcounts, self.minwcount)
        
        annotations_train['questions'] = self.encode_questions(annotations_train['questions'], word_to_wid)
        if self.proc_split == 'train' and self.has_valset:
            annotations_val['questions'] = self.encode_questions(annotations_val['questions'], word_to_wid)
        if self.has_testset:
            annotations_test['questions'] = self.encode_questions(annotations_test['questions'], word_to_wid)
        
        annotations_train['questions'] = self.encode_answers(annotations_train['questions'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            annotations_val['questions'] = self.encode_answers(annotations_val['questions'], ans_to_aid)
        if self.has_testset_anno:
            annotations_test['questions'] = self.encode_answers(annotations_test['questions'], ans_to_aid)
        
        if self.has_answers_occurence:
            annotations_train['questions'] = self.add_answers_occurence(annotations_train['questions'], ans_to_aid)

        Logger()('Save processed datasets to {}'.format(self.subdir_processed))
        os.system('mkdir -p '+self.subdir_processed)
        if self.has_testdevset:
            is_qid_testdev = {item['question_id']:True for item in testdev['questions']}
            torch.save(is_qid_testdev, self.path_is_qid_testdev)
        torch.save(wid_to_word, self.path_wid_to_word)
        torch.save(word_to_wid, self.path_word_to_wid)
        torch.save(aid_to_ans, self.path_aid_to_ans)
        torch.save(ans_to_aid, self.path_ans_to_aid)
        torch.save(annotations_train, self.path_trainset)
        if self.proc_split == 'train' and self.has_valset:
            torch.save(annotations_val, self.path_valset)
        if self.has_testset:
            torch.save(annotations_test, self.path_testset)

    def tokenize_answers(self, annotations):
        Logger()('Example of modified answers after preprocessing:')
        for i, ex in enumerate(annotations):
            s = ex['answer']
            if self.nlp == 'nltk':
                ex['answer'] = " ".join(word_tokenize(str(s).lower()))
            elif self.nlp == 'mcb':
                ex['answer'] = " ".join(tokenize_mcb(s))
            else:
                ex['answer'] = " ".join(tokenize(s))
            if i < 10: Logger()('{} became -> {} <-'.format(s,ex['answer']))
            if i>0 and i % 1000 == 0:
                sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(annotations), i*100.0/len(annotations)) )
                sys.stdout.flush() 
        return annotations

    def add_image_names(self, dataset):
        
        # names in the annotations, will be used to get items from the dataset
        self.img_names = [s['image'] for s in self.annotations]
        # load features
        self.features = FeaturesDataset(config['images']['path_features'], config['images']['mode'])
        
        for q in dataset['questions']:
            q['image_name'] = 'COCO_%s_%012d.jpg'%(dataset['data_subtype'],q['image_id'])
        return dataset

    def add_answer(self, annotations, answers_anw_to_id):
        for item in annotations:
            first_candidate = "unanswerable"
            # print(item.keys())
            # print(item['image'])
            # print(item['question'])
            # print(item['answers'])
            # print(item['answer_type'])
            # print(item['answerable'])
            # print(item['image_id'])
            # print("----------------------------------------------")
            for a in item['answers']:
                if a['answer'] in answers_anw_to_id.keys() and a['answer_confidence'] == 'yes':
                    first_candidate = a['answer']
                    break
            item['answer'] = first_candidate
            # print(item['answers'])
            # print(answers_anw_to_id.keys())
            item['answer_id'] = answers_anw_to_id[first_candidate]
        return annotations

    def top_answers(self, annotations, nans):
        counts = {}
        for item in tqdm(annotations):
            ans = item['answer'] 
            counts[ans] = counts.get(ans, 0) + 1

        cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
        Logger()('Top answer and their counts:')
        for i in range(20):
            Logger()(cw[i])

        vocab = []
        for i in range(nans):
            vocab.append(cw[i][1])
        Logger()('Number of answers left: {} / {}'.format(len(vocab), len(cw)))
        return vocab[:nans]

    def annotations_in_top_answers(self, questions, top_answers):
        
        # new_anno = []
        new_ques = []
        # if len(annotations) != len(questions): # sanity check
        #     raise ValueError()
        for i in tqdm(range(len(questions))):
            # print(questions[i].keys())
            # print(questions[i]['answerable'])
            # print(questions[i]['answer'])
            # print(len(top_answers))
            # print(new_ques['f'])
            # print(new_ques['f'])
            # print(questions[i]['answer'])
            # print(new_ques['f'])
            if self.annotations_in_top_answers_only:
                # print("i'm here ---------------------")
                if questions[i]['answer'] in top_answers and questions[i]['answerable'] == 1 and questions[i]['answer'] != 'unanswerable':
                    new_ques.append(questions[i])
            else:
                if questions[i]['answerable'] == 1 and questions[i]['answer'] != 'unanswerable':
                    new_ques.append(questions[i])
        Logger()('Number of examples reduced from {} to {}'.format(
            len(questions), len(new_ques)))
        # print(new_ques['f'])
        return new_ques

    def tokenize_questions(self, questions, nlp):
        Logger()('Tokenize questions')
        if nlp == 'nltk':
            from nltk.tokenize import word_tokenize
        for item in tqdm(questions):
            # print(item.keys())
            # print(item['question'])
            # print(item['questionn'])
            ques = item['question']
            if nlp == 'nltk':
                item['question_tokens'] = word_tokenize(str(ques).lower())
            elif nlp == 'mcb':
                item['question_tokens'] = tokenize_mcb(ques)
            else:
                item['question_tokens'] = tokenize(ques)
        return questions

    def top_words(self, questions, minwcount):
        wcounts = {}
        for item in questions:
            for w in item['question_tokens']:
                wcounts[w] = wcounts.get(w, 0) + 1
        cw = sorted([(count,w) for w, count in wcounts.items()], reverse=True)
        Logger()('Top words and their wcounts:')
        for i in range(20):
            Logger()(cw[i])

        total_words = sum(wcounts.values())
        Logger()('Total words: {}'.format(total_words))
        bad_words = [w for w in sorted(wcounts) if wcounts[w] <= minwcount]
        vocab = [w for w in sorted(wcounts) if wcounts[w] > minwcount]
        bad_count = sum([wcounts[w] for w in bad_words])
        Logger()('Number of bad words: {}/{} = {:.2f}'.format(len(bad_words), len(wcounts), len(bad_words)*100.0/len(wcounts)))
        Logger()('Number of words in vocab would be {}'.format(len(vocab)))
        Logger()('Number of UNKs: {}/{} = {:.2f}'.format(bad_count, total_words, bad_count*100.0/total_words))
        vocab.append('UNK')
        return vocab, wcounts

    def merge_annotations_with_questions(self, ann, ques):
        for key in ann:
            if key not in ques:
                ques[key] = ann[key]
        return ques

    def insert_UNK_token(self, questions, wcounts, minwcount):
        for item in questions:
            item['question_tokens_UNK'] = [w if wcounts.get(w,0) > minwcount else 'UNK' for w in item['question_tokens']]
        return questions

    def encode_questions(self, questions, word_to_wid):
        for item in questions:
            item['question_wids'] = [word_to_wid[w] for w in item['question_tokens_UNK']]
            q_id = ""
            for i in item['question_wids'] :
                q_id += str(i)
            item['question_id'] = str(item['image_id']) + q_id
            # print(item.keys())
            # print(item['image_id'])
            # print(item['question_wids'])
            # print(item['question_id'])
            # print(item['question_widsss'])
        return questions

    def encode_answers(self, annotations, ans_to_aid):
        for item in annotations:
            # associate to examples when answer is not in classes dictionnary
            # the class_id of the less occuring answer
            # Warning: the accuracy may be a bit wrong (to use only during validation)
            item['answer_id'] = ans_to_aid.get(item['answer'], len(ans_to_aid)-1)
        return annotations

    def add_answers_occurence(self, annotations, ans_to_aid):
        # for samplingans during training
        for item in annotations:
            item['answers_word'] = []
            item['answers_id'] = []
            item['answers_count'] = []
            # print(item.keys())
            answers = [a['answer'] for a in item['answers']]
            for ans, count in dict(Counter(answers)).items():
                if ans in ans_to_aid:
                    item['answers_word'].append(ans)
                    item['answers_id'].append(ans_to_aid[ans])
                    item['answers_count'].append(count)
                    # print("cccccccccccccccccc", count)
                # print("cccccccccccccccccc", count)
            # if len(item['answers_count']) == 0:
            #     print(answers)
            #     print(item['answer'])
            #     print(answers['i'])
                    # first_candidate = "unanswerable"
                
        return annotations

    #############################################################################""
    # Preprocessing on a list dataset (vqa2+vgenome setup)

    def sync_from(self, dataset):
        Logger()('Sync {}.{} from {}.{}'.format(
            self.__class__.__name__,
            self.split,
            dataset.__class__.__name__,
            dataset.split))

        if 'annotations' in self.dataset:
            Logger()('Removing triplets with answer not in dict answer')
            list_anno = []
            list_ques = []
            for i in tqdm(range(len(self))):
                ques = self.dataset['questions'][i]
                answer = self.dataset['annotations'][i]['answer']
                if answer in dataset.ans_to_aid:
                    # sync answer_id
                    anno = self.dataset['annotations'][i]
                    anno['answer_id'] = dataset.ans_to_aid[answer]

                    list_anno.append(anno)
                    list_ques.append(ques)
            self.dataset['annotations'] = list_anno
            self.dataset['questions'] = list_ques
            Logger()('{} / {} remaining questions after filter_from'.format(len(list_ques), len(self.dataset['questions'])))

        # sync question_wids
        Logger()('Sync question_wids')
        for i in tqdm(range(len(self))):
            ques = self.dataset['questions'][i]
            for j, token in enumerate(ques['question_tokens']):
                if token in dataset.word_to_wid:
                    ques['question_tokens_UNK'][j] = token
                    ques['question_wids'][j] = dataset.word_to_wid[token]
                else:
                    ques['question_tokens_UNK'][j] = 'UNK'
                    ques['question_wids'][j] = dataset.word_to_wid['UNK']

        # sync dict word, dict ans
        self.word_to_wid = dataset.word_to_wid
        self.wid_to_word = dataset.wid_to_word
        self.ans_to_aid = dataset.ans_to_aid
        self.aid_to_ans = dataset.aid_to_ans

