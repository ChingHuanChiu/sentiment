import sys

sys.path.extend(['./', './algo/roberta'])

from transformers import BertTokenizer, TFBertModel
from typing import Generator

import pandas as pd
import os
import tensorflow as tf
import logging

logging.disable(30)


class RoBertaFineTune(tf.keras.Model):
    def __init__(self, freeze: bool):
        super(RoBertaFineTune, self).__init__()
        self.freeze = freeze

        self.bert = TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext', output_attentions=True)

        self.dense_128 = tf.keras.layers.Dense(128, activation='relu')

        self.dense_64 = tf.keras.layers.Dense(64, activation='relu')

        self.dense = tf.keras.layers.Dense(2, activation='softmax')
        
#         self.dropout = tf.keras.layers.Dropout(0.2)
        self.dropout_2 = tf.keras.layers.Dropout(0.1)

    @tf.function
    def call(self, inputs):
        inp_dict = {"input_ids": inputs[0], "attention_mask": inputs[2], "token_type_ids": inputs[1]}
        if self.freeze is True:
            self.bert.trainable = False

        model = self.bert(inp_dict)
        pooler = model.pooler_output

        x = self.dense_128(pooler)
#         x = self.dropout(dense_128)
        x = self.dense_64(x)
        x = self.dropout_2(x)
        out = self.dense(x)

        return out

    @staticmethod
    def predict(data_loader: Generator, model, return_logistis_and_pooler=False):
        """
        TODO: add the max_seq_length、batch_size to config
        """

        predict_BTL = data_loader

        X_data = next(predict_BTL.__iter__())[0]  # wordpiece_token, segement_token, attention_mask

        y_pred_logitis = model(X_data)
        
        #         y_pred = tf.squeeze(tf.where(y_pred_logitis > threshold, 1, 0)).numpy()
        y_pred = tf.math.argmax(y_pred_logitis, 1)
        if return_logistis_and_pooler:
            pooler_out = model.bert(X_data).pooler_output
            return y_pred, y_pred_logitis, pooler_out
        else:
            return y_pred

    @classmethod
    def load_model(cls, checkpoint_path, latest=True):
        print('.....model is loading')
        model_ = cls(freeze=False)
        checkpoint = tf.train.Checkpoint(Model=model_)
        if latest is True:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
        else:
            checkpoint.restore(checkpoint_path)
        print('You got it')
        return model_


def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
