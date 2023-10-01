# HiPA: Hierarchical Reasoning Based on Perception Action Cycle for Visual Question Answering

Recent visual question answering (VQA) frameworks employ different attention modules to derive a correct answer. The concept of attention is heavily established in human cognition, which led to its magnificent success in deep neural networks. In this study, we aim to consider a VQA framework that utilizes human biological and psychological concepts to achieve a good understanding of vision and language modalities. In this view, we introduce a hierarchical reasoning method based on the perception action cycle (HIPA) framework to tackle VQA tasks. The perception action cycle (PAC) explains how humans learn about and interact with their surrounding world. The proposed framework integrates the reasoning process of multi-modalities with the concepts introduced in PAC in multiple phases. It comprehends the visual modality through three phases of reasoning: object-level attention, organization, and interpretation. In addition, it comprehends the language modality through word-level attention, interpretation, and conditioning. Subsequently, vision and language modalities are interpreted dependently in a cyclic and hierarchical way throughout the entire framework. For further assessment of the generated visual and language features, we argue that image-question pairs of the same answer ought to eventually have similar visual and language features. As a result, we conduct visual and language feature evaluation experiments using metrics such as the standard deviation of cosine similarity and of Manhattan distance. We show that employing PAC in our framework improves the standard deviation compared with other VQA frameworks. For further assessment, we also test the novel proposed HIPA on the visual relationship detection (VRD) task. The proposed method achieves state-of-the-art results on the TDIUC and VRD datasets and obtains competitive results on the VQA 2.0 dataset.

The model is trained on 63 GB GPU. One epoch takes from 15 to 20 mins. All models are trained to 25 epochs. Accuracy metrics codes are provided by datasets creators.

#### Summary

* [Installation](#installation)
    * [Python 3 & Anaconda](#1-python-3--anaconda)
    * [Download datasets](#3-download-datasets)
* [Quick start](#quick-start)
    * [Train a model](#train-a-model)
    * [Evaluate a model](#evaluate-a-model)
* [Reproduce results](#reproduce-results)
    * [VQA2](#vqa2-dataset)
    * [TDIUC](#tdiuc-dataset)
    * [VRD](#vrd-dataset)



## Installation

### 1. Python 3 & Anaconda

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://www.continuum.io/downloads). Then, you can create an environment.

### 2. As standalone project

```
conda create --name hipa python=3.7
source activate hipa
cd HiPA Framework
pip install -r requirements.txt
```

### 3. Download datasets

Download annotations, images and features for VQA experiments:
```
bash HiPA/datasets/scripts/download_vqa2.sh
bash HiPA/datasets/scripts/download_vgenome.sh
bash HiPA/datasets/scripts/download_tdiuc.sh
bash HiPA/datasets/scripts/download_vrd.sh
```

**Note:** The features have been extracted from a pretrained Faster-RCNN with caffe. We don't provide the code for pretraining or extracting features for now.


## Quick start

### Train a model

You can train our best model on VQA2 by running:
```
python -m run -o HiPA/options/vqa2/HiPA.yaml
```
Then, several files are going to be created in `logs/vqa2/HiPA`:
- ckpt_last_engine.pth.tar (checkpoints of last epoch)
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_engine.pth.tar (checkpoints of best epoch)
- ckpt_best_eval_epoch.accuracy_top1_model.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_optimizer.pth.tar


### Evaluate a model

At the end of the training procedure, you can evaluate your model on the testing set. 
```
python -m run -o logs/vqa2/HiPA/options.yaml --exp.resume best_accuracy_top1 --dataset.train_split --dataset.eval_split test --misc.logs_name test
```

## Reproduce results

### VQA2 dataset

VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
-265,016 images (COCO and abstract scenes)
-At least 3 questions (5.4 questions on average) per image
-10 ground truth answers per question
-3 plausible (but likely incorrect) answers per question
-Automatic evaluation metric

VQA 2.0 has three splits. The training split contains 82,783 images with 443,757 questions. The validation split has 40,504 images with 214,354 questions. The test-standard split contains 81,434 images with 447,793 questions.

### TDIUC dataset


The TDIUC dataset has 167,437 images, 1,654,167 questions, 12 question types and 1,618 unique answers.
Question Type - Number of Questions - Number of Unique Answers: 
Scene Recognition - 66,706 - 83
Sport Recognition - 31,644 - 12
Color Attributes - 195,564 - 16
Other Attributes - 28,676 - 625
Activity Recognition - 8,530 - 13
Positional Reasoning - 38,326 - 1,300
Sub. Object Recognition - 93,555 - 385
Absurd - 366,654 - 1
Utility/Affordance - 521 - 187
Object Presence - 657,134 - 2
Counting - 164,762 - 16
Sentiment Understanding - 2,095 - 54
Grand Total - 1,654,167 - 1,618

#### Training and evaluation (train/val/test)

The full training set is split into a trainset and a valset. At the end of the training, we evaluate our best checkpoint on the testset. The TDIUC metrics are computed and displayed at the end of each epoch. They are also stored in `logs.json` and `logs_test.json`.


```
python -m run -o HiPA/options/tdiuc/HiPA.yaml --exp.dir logs/tdiuc/HiPA

```

### VRD dataset


The Visual Relationship Dataset (VRD) contains 4000 images for training and 1000 for testing annotated with visual relationships. Bounding boxes are annotated with a label containing 100 unary predicates. These labels refer to animals, vehicles, clothes and generic objects. Pairs of bounding boxes are annotated with a label containing 70 binary predicates. These labels refer to actions, prepositions, spatial relations, comparatives or preposition phrases. The dataset has 37993 instances of visual relationships and 6672 types of relationships. 1877 instances of relationships occur only in the test set and they are used to evaluate the zero-shot learning scenario.


#### Training and evaluation (train/val/test)

The full training set is split into a trainset and a valset. At the end of the training, we evaluate our best checkpoint on the testset. 


```
python -m run -o HiPA/options/vrd/HiPA.yaml --exp.dir logs/vrd/HiPA

```


