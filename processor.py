import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
# from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle

from typing import List, Tuple

# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


class Preprocessor:
    """
    For sentiment classification mission, {'negative': 0, 'positive': 1}
    """

    def __init__(self, data: pd.DataFrame, mode, imbalance_sample, tokenizer):
        self.data = data.copy()
        if imbalance_sample is True:
            print('doing random oversampling')
            train_X, train_y = self.data.content.values.reshape(-1, 1), self.data.label.values.reshape(-1, 1)
            ros = RandomOverSampler(sampling_strategy='minority')
            train_X, train_y = ros.fit_resample(train_X, train_y)
            # Need to assert that train_X[:, 0] must be content!!!!!
            self.data = shuffle(pd.DataFrame({'content': train_X[:, 0], 'label': train_y})).reset_index(drop=True)

        self.mode = mode

        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        """
        :rtype: Tuple(List...)
        :return: tokens_tensors  : (batch_size, max_seq_len_in_batch)
                 segments_tensors: (batch_size, max_seq_len_in_batch)
                 masks_tensors   : (batch_size, max_seq_len_in_batch)
                 label_ids       : (batch_size)
        """

        tokenize = self.tokenizer.encode_plus(self.data.content.values[idx])

        wordpiece_token = tokenize['input_ids']

        segments_token = tokenize['token_type_ids']

        attention_mask = tokenize['attention_mask']

        if self.mode == 'train':
            label_tensor = self.data.label.values[idx]
            #             label_tensor = to_categorical(label_tensor, num_classes=2)
            return wordpiece_token, segments_token, attention_mask, label_tensor
        else:
            return wordpiece_token, segments_token, attention_mask


class DataLoader(Preprocessor):

    def __init__(self, data, MAX_SEQUENCE_LENGTH, batch_size, imbalance_sample, tokenizer=None, mode='train'):
        super(DataLoader, self).__init__(data, mode, imbalance_sample, tokenizer)
        self.batch_size = batch_size
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.mode = mode

    def __iter__(self):

        start = 0
        end = self.batch_size
        while start <= len(self.data) and start != end:

            output: List[Tuple] = [self[i] for i in range(start, end)]

            wordpiece_token, segements_token, attention_mask, label_token = [], [], [], []
            for t in output:
                wordpiece_token.append(t[0])
                segements_token.append(t[1])
                attention_mask.append(t[2])
                if self.mode == 'train':
                    label_token.append(t[3])

            # Padding
            res = tuple()
            for token in [wordpiece_token, segements_token, attention_mask]:
                t = keras.preprocessing.sequence.pad_sequences(token, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
                t = tf.convert_to_tensor(t)
                res += (t,)

            res = ((res), tf.convert_to_tensor(label_token),) if label_token else res

            yield res  # wordpiece_token, segement_token, attention_mask, or label_token
            start += self.batch_size
            end = len(self.data) if start + self.batch_size > len(self.data) else start + self.batch_size


