from allennlp.data.fields import *
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.nn.util import get_text_field_mask
from allennlp.data.tokenizers import Token
from allennlp.models import BasicClassifier, Model
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, Average, Metric
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics.metric import Metric
from allennlp.nn import util

from typing import *
from overrides import overrides
import jieba
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import cv2 as cv
import os

torch.manual_seed(123)

def process_image(img, min_side=224):
    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv.resize(img, (new_w, new_h))
    top, bottom, left, right = 0, min_side - new_h, 0, min_side - new_w

    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255, 255, 255])

    return pad_img


@DatasetReader.register("s2s_manual_reader")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        sub_dict_path = "data/sub_label_dict.pk"  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        self.all_points = ['干电池', '串联电路', '并联电路', '欧姆定律', '电能公式', '电功率', '焦耳定律', '短路']

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)

        image = image / 255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image

        fields['image'] = ArrayField(img_rgb)

        s_token = self._tokenizer.tokenize(' '.join(sample['token_list']))
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
        t_token = self._tokenizer.tokenize(' '.join(sample['manual_program']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
        fields['source_nums'] = MetadataField(sample['number'])

        # fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['answer_number'] = MetadataField(sample['answer_number'])
        # fields['label'] = MetadataField(sample['label'])
        fields['answer'] = MetadataField(sample['answer'])

        type = self.subset_dict[sample['id']]
        fields['type'] = MetadataField(type)
        fields['data_id'] = MetadataField(sample['id'])

        equ_list = []

        equ = sample['manual_program']
        equ_token = self._tokenizer.tokenize(' '.join(equ))
        equ_token.insert(0, Token(START_SYMBOL))
        equ_token.append(Token(END_SYMBOL))
        equ_token = TextField(equ_token, self._source_token_indexer)
        equ_list.append(equ_token)

        fields['equ_list'] = ListField(equ_list)
        fields['manual_program'] = MetadataField(sample['manual_program'])

        point_label = np.zeros(8,
                               np.float32)
        exam_points = sample['formal_point']
        for point in exam_points:
            point_id = self.all_points.index(point)
            point_label[point_id] = 1
        fields['point_label'] = ArrayField(np.array(point_label))

        return Instance(fields)
