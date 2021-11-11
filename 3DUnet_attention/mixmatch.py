import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
#import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six

import tensorflow as tf
# whole lots of whatever error ended up with self._traceback = tf_stack.extract_stack()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.executing_eagerly()

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_number
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu
import config
from model import ( unet3d, unet3d_attention, Loss )
from data_sampler import (get_train_dataflow, get_eval_dataflow, get_test_dataflow)
from eval import (eval_brats, pred_brats, segment_one_image)

from MedPy import compute_f1_hdd_3d

# import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu

def get_model_output_names():
    ret = ['final_probs', 'final_pred']
    return ret

def get_model(modelType="training", inference_shape=config.INFERENCE_PATCH_SIZE):
    return Unet3dModel(modelType=modelType, inference_shape=inference_shape)

manual_epoch = 1

class Unet3dModel(ModelDesc):
    def __init__(self, modelType="training", inference_shape=config.INFERENCE_PATCH_SIZE):
        self.modelType = modelType
        self.inference_shape = inference_shape
        print(self.modelType)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=config.BASE_LR, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        #opt = tf.train.RMSPropOptimizer(lr)
        #opt = tf.train.AdamOptimizer()
        return opt
        
    def preprocess(self, image):
        # transform to NCHW
        return tf.transpose(image, [0, 4, 1, 2, 3])

    def inputs(self):
        S = config.PATCH_SIZE
        if self.modelType == 'training':
            if config.MIXUP:
                ret = [
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'image'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'weight'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'label')]
            else:
                ret = [
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'image'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'weight'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'label')]
        else:
            S = self.inference_shape
            ret = [
                tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'image')]
        return ret

    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        if is_training:
            image, weight, label = inputs
        else:
            image = inputs[0]
        
        if config.MULTI_LOSS == True:
            C12, C345, featuremap = unet3d_attention('unet3d_attention', image)
        else:     
            featuremap = unet3d_attention('unet3d_attention', image)
        
        if is_training:
            loss = Loss(featuremap, weight, label)

            wd_cost = regularize_cost(
                    '(?:unet3d_attention)/.*kernel',
                    #'(?:unet3d)/.*kernel',
                    l2_regularizer(1e-5), name='wd_cost')

            if config.MULTI_LOSS == True:
                loss12 = Loss(C12, weight, label)
                loss345 = Loss(C345, weight, label)
                total_cost = tf.add_n([loss12, loss345, loss, wd_cost], 'total_cost')
            else:
                total_cost = tf.add_n([loss, wd_cost], 'total_cost')
            
            add_moving_summary(total_cost, wd_cost)

            if config.NUM_CLASS == 1:
                final_probs = tf.math.sigmoid(featuremap, name="final_probs")
                final_pred = tf.argmax(final_probs, axis=-1, name="final_pred")
            else:
                final_probs = tf.nn.softmax(featuremap, name="final_probs")
                #final_probs = tf.identity(featuremap, name="final_probs")
                final_pred = tf.argmax(final_probs, axis=-1, name="final_pred")
            return total_cost
        else:
            if config.NUM_CLASS == 1:
                final_probs = tf.math.sigmoid(featuremap, name="final_probs")
                final_pred = tf.argmax(final_probs, axis=-1, name="final_pred")
            else:
                final_probs = tf.nn.softmax(featuremap, name="final_probs")
                #final_probs = tf.identity(featuremap, name="final_probs")
                final_pred = tf.argmax(final_probs, axis=-1, name="final_pred")

class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], get_model_output_names())
        self.df = get_eval_dataflow()
    
    def _eval(self):
        dice, f1, hdd = eval_brats(self.df, lambda img: segment_one_image(img, [self.pred], is_online=True))
        for k, v in dice.items():
            self.trainer.monitors.put_scalar('val_dice_'+k, v)
        for k, v in hdd.items():
            self.trainer.monitors.put_scalar('val_hdd_'+k, v)


    def _trigger_epoch(self):
        if self.epoch_num > 0 and self.epoch_num % config.EVAL_EPOCH == 0:
            self._eval()

def offline_evaluate(pred_func, output_file):
        df = get_eval_dataflow()
        if config.DYNAMIC_SHAPE_PRED:    
            eval_brats(
                df, lambda img: segment_one_image_dynamic(img, pred_func))
        else:
            eval_brats(
                df, lambda img: segment_one_image(img, pred_func))

def offline_pred(pred_func, output_file):
    df = get_test_dataflow()
    pred_brats(
        df, lambda img: segment_one_image(img, pred_func))


class MixMatch(Unet3dModel):
    def augment(self, x, l, beta, **kwargs):
        assert 0, 'Do not call lol @todo'
    
    def guess_label(self, y, classifier, T, **kwargs):
        del kwargs
        logits_y = [classifier(yi, training=True) for yi in y]
        logits_y = tf.concat(logits_y, 0)
        # Compute predicted probability distribution py.
        p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)
        # Compute the target distribution.
        p_target = tf.pow(p_model_y, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, nu=2, mixmode='xxy.yxy', **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        augment = MixMode(mixmode)
        classifier = functools.partial(self.classifier, **kwargs)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        lx = tf.one_hot(l_in, self.nclass)
        xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        del xy, labels_xy

        batches = layers.interleave([x] + y, batch)
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = [classifier(batches[0], training=True)]
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        for batchi in batches[1:]:
            logits.append(classifier(batchi, training=True))
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_l2u = tf.reduce_mean(loss_l2u)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/l2u', loss_l2u)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(batches[0], training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to all availalbe ones', default='0')
    parser.add_argument('--load', help='load model for evaluation or training')
    parser.add_argument('--logdir', help='log directory', default='train_log/unet3d')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', action='store_true', help="Run evaluation")
    parser.add_argument('--predict', action='store_true', help="Run prediction")
    parser.add_argument('--fold', type=int, help="cross validation k fold")
    parser.add_argument('--multi_loss', type=bool, default=False, help="multi-loss regression or not")
    args = parser.parse_args()
   
    config.MULTI_LOSS = args.multi_loss
    if args.fold is not None:
        config.FOLD = args.fold
        args.logdir = "train_log/unet3d_2020_5fold-" + str(args.fold)

    # args.logdir = "train_log/unet3d_2020_5fold-" + str(args.fold) + "_multi_loss"

    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if tf.test.gpu_device_name(): 
            print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        print(args.gpu)
        input('wait a second')

    if args.visualize or args.evaluate:
        if config.DYNAMIC_SHAPE_PRED:
            def get_dynamic_pred(shape):
                return OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference", inference_shape=shape),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            offline_evaluate([get_dynamic_pred], args.evaluate)
        elif config.MULTI_VIEW:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred1 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_sa/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred2 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_cr/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            offline_evaluate([pred, pred1, pred2], args.evaluate)
        else:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader(args.load) if args.load else None,
                    input_names=['image'],
                    output_names=get_model_output_names()))
            # autotune is too slow for inference
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            #assert args.load
            offline_evaluate([pred], args.evaluate)
    
    elif args.predict:
        if config.MULTI_VIEW:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred1 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_sa/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred2 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_cr/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            # autotune is too slow for inference
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            offline_pred([pred, pred1, pred2], args.evaluate)
        else:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            # autotune is too slow for inference
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            assert args.load
            offline_pred([pred], args.evaluate)

        
    else:
        logger.set_logger_dir(args.logdir)
        factor = get_batch_factor()
        stepnum = config.STEP_PER_EPOCH


        cfg = AutoResumeTrainConfig(
            model=get_model(),
            data=QueueInput(get_train_dataflow()),
            callbacks=[
                PeriodicCallback(
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    every_k_epochs=10),
                ScheduledHyperParamSetter('learning_rate', 
                    [(40, config.BASE_LR*0.1),
                    (60, config.BASE_LR*0.01),
                    (80, config.BASE_LR*0.01),
                    (140, config.BASE_LR*0.001),
                    (160, config.BASE_LR*0.0001),
                    (180, config.BASE_LR*0.00001)]
                ),
                EvalCallback(),
                #GPUUtilizationTracker(),
                #PeakMemoryTracker(),
                EstimatedTimeLeft()
            ],
            steps_per_epoch=stepnum,
            max_epoch=config.MAX_EPOCH,
            session_init=get_model_loader(args.load) if args.load else None,
        )
        # nccl mode gives the best speed
        trainer = SimpleTrainer()
        # trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), mode='nccl')
        launch_train_with_config(cfg, trainer)
        
