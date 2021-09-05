from abc import ABCMeta, abstractmethod
from typing import List, Generator
from tqdm import tqdm

import math
import tensorflow as tf


class TrainBase(metaclass=ABCMeta):

    @abstractmethod
    def train_step(self, X_batch, y_batch, optimizer, loss_recoder: List):
        pass

    @abstractmethod
    def setting_optimizer(self, decay_steps):

        pass

    def start_to_train(self, train_data_loader: Generator,
                       epochs,
                       checkpoint_path,
                       tensorboard_path: dict,
                       validation_data_loader: Generator = None):

        BATCH_LENGHT = math.ceil(train_data_loader.data.shape[0] / train_data_loader.batch_size)

        # saving manager
        checkpoint = tf.train.Checkpoint(Model=self.model)
        #         manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=10)

        # Initialize Metric
        TrainMetric = self.metric()

        writer = tf.summary.create_file_writer(tensorboard_path['train'])

        optimizer = self.setting_optimizer(decay_steps=epochs)
        for epoch in range(epochs):
            print(optimizer.lr(optimizer.iterations))
            print('==========================')
            print(f'EPOCH {epoch}')
            print('==========================')

            loss_recoder = []

            #             tf.summary.trace_on(graph=True)

            for step, (X_batch_train, y_batch_train) in tqdm(enumerate(train_data_loader), total=BATCH_LENGHT):
                logits, loss_recoder = self.train_step(X_batch=X_batch_train,
                                                       y_batch=y_batch_train,
                                                       optimizer=optimizer,
                                                       loss_recoder=loss_recoder)

                TrainMetric.calculate_metric(y_batch_train, logits=logits)
                if step % 1000 == 0:
                    #                     manager.save(checkpoint_number=step)
                    checkpoint.save(f'{checkpoint_path}/ckpt-epoch:{epoch}')

            epoch_loss = sum(loss_recoder) / len(loss_recoder)

            print(f"training loss is {epoch_loss}")

            for name, result in TrainMetric.get_result.items():
                print(f'Training {name} over epoch : {float(result)}')
            print('============================================================')

            #             with writer.as_default():
            #                 tf.summary.trace_export(name="my_trace",step=0)

            TrainTB = TensorBoard(writer=writer,
                                  model=self.model)
            TrainTB.start_to_write(metrics_result=TrainMetric.get_result,
                                   step=epoch,
                                   loss=epoch_loss,
                                   histogram=True,
                                   optimizer=optimizer)

            TrainMetric.reset()

            if validation_data_loader is not None:
                self._validation(validation_data_loader=validation_data_loader,
                                 val_logdir=tensorboard_path['validation'],
                                 ep=epoch)

    def _validation(self, validation_data_loader, val_logdir, ep):
        val_loss_recoder = []

        ValMetric = self.metric()
        for x_batch_val, y_batch_val in validation_data_loader:
            val_logits = self.model(x_batch_val)

            val_loss_value = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_batch_val, val_logits)
            val_loss_recoder.append(val_loss_value.numpy())

            ValMetric.calculate_metric(y_batch_val, logits=val_logits)

        val_epoch_loss = sum(val_loss_recoder) / len(val_loss_recoder)
        print(f" loss of validation is {val_epoch_loss}")

        for name, result in ValMetric.get_result.items():
            print(f'Validation {name} over epoch : {float(result)}')
        print('============================================================')

        val_writer = tf.summary.create_file_writer(val_logdir)

        ValTB = TensorBoard(writer=val_writer, model=self.model)

        ValTB.start_to_write(metrics_result=ValMetric.get_result,
                             step=ep,
                             loss=val_epoch_loss,
                             histogram=False)

        ValMetric.reset()


class IMetric(metaclass=ABCMeta):

    @abstractmethod
    def calculate_metric(self, y_batch, logits):
        pass

    @property
    def get_result(self) -> dict:
        """
        return a dictionary of result of metrics
        """
        result = dict()
        for name, metric in self.__dict__.items():
            result[name] = metric.result()

        return result

    def reset(self):

        for metric in self.__dict__.values():
            metric.reset_states()


class TensorBoard:

    def __init__(self, writer, model):
        self.model = model
        self.writer = writer

    def start_to_write(self, metrics_result, step, loss=None, histogram=False, optimizer=None):
        with self.writer.as_default():
            if loss is not None:
                tf.summary.scalar("loss", loss, step=step)

            if histogram is True:
                for layer in self.model.layers:
                    try:
                        for w in layer.weights:
                            tf.summary.histogram(w.name, w, step=step)
                    except:
                        tf.summary.histogram(layer.name, layer.weights, step=step)

            for name, result in metrics_result.items():
                tf.summary.scalar(name, result, step=step)

            try:
                if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                    current_lr = optimizer.lr(optimizer.iterations)
                else:
                    current_lr = optimizer.lr
                #             print(current_lr)
                tf.summary.scalar('learning rate', data=current_lr, step=step)
            except:
                pass
