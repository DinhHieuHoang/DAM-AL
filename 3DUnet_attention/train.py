from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse, shutil, itertools, tqdm, json, six, sys
import numpy as np
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
from model import ( unet3d_attention, Loss )
from data_sampler import (get_train_dataflow, get_eval_dataflow, get_test_dataflow)
from eval import (eval_brats, pred_brats, segment_one_image)
from MedPy import compute_f1_hdd_3d
import config

# import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

global args 

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.basename(args.logdir) + ".txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def get_book_variable_module_name(module_name): # __author__ = 'spouk' from stackoverflow
    module = globals().get(module_name, None)
    book = {}
    if module:
        book = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    return book

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
        print(self.modelType, self.inference_shape, config.INFERENCE_PATCH_SIZE)
        self.inference_shape = config.INFERENCE_PATCH_SIZE

    def optimizer(self):
        #opt = tf.train.RMSPropOptimizer(lr)
        if args.adam or args.warmup_lr:
            lr = tf.get_variable('learning_rate', initializer=config.BASE_LR_ADAM, trainable=False)
            tf.summary.scalar('learning_rate', lr)
            opt = tf.train.AdamOptimizer()
        else:
            lr = tf.get_variable('learning_rate', initializer=config.BASE_LR_ADAM, trainable=False)
            tf.summary.scalar('learning_rate', lr)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
        
    def preprocess(self, image):
        # transform to NCHW
        return tf.transpose(image, [0, 4, 1, 2, 3])

    def inputs(self):
        S = config.PATCH_SIZE
        if config.DATASET == 'iseg':
            modalities = 2
        else:
            modalities = 4
        if self.modelType == 'training':
            if config.MIXUP:
                ret = [
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], modalities), 'image'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'weight'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'label')]
            elif config.BOUNDARY_LOSS or config.BOUNDARY_FOCAL:
                ret = [
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], modalities), 'image'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'weight'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'label'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 3), 'sdm')]
            else:
                ret = [
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], modalities), 'image'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'weight'),
                    tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'label')]
        else:
            S = self.inference_shape
            ret = [
                tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], modalities), 'image')]
        return ret

    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        distance_map =None
        if is_training:
            image, weight, label = inputs[0:3]
            if config.BOUNDARY_LOSS or config.BOUNDARY_FOCAL:
                distance_map = inputs[3]
        else:
            image = inputs[0]
        
        if config.MULTI_LOSS == True:
            C12, C345, featuremap = unet3d_attention('unet3d_attention', image)
        else:     
            featuremap = unet3d_attention('unet3d_attention', image)
        
        if is_training:
            loss = Loss(featuremap, weight, label, distance_map)

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
        print('@dghan hello setup graph')
        # global previous_predictor
        # previous_predictor = self.pred
        self.pred = self.trainer.get_predictor(
            ['image'], get_model_output_names())
        self.df = get_eval_dataflow()
    
    def _eval(self):
        dice, hdd, mhdd, asd = eval_brats(self.df, lambda img: segment_one_image(img, [self.pred], is_online=True), save_nii=False, no_f1_hdd=args.no_f1_hdd)
        label = list(config.idx2class.values())
        for k, v in dice.items():
            self.trainer.monitors.put_scalar('val_dice_'+k, v)
        print('val_dice', (dice[label[1]] + dice[label[2]] + dice[label[3]])/3.0)
        self.trainer.monitors.put_scalar('val_dice', (dice[label[1]] + dice[label[2]] + dice[label[3]])/3.0)
        if args.no_f1_hdd:
            for k, v in hdd.items():
                self.trainer.monitors.put_scalar('val_hdd_'+k, v)
            print('val_hdd', (hdd[label[1]] + hdd[label[2]] + hdd[label[3]])/3.0)
            self.trainer.monitors.put_scalar('val_hdd', (hdd[label[1]] + hdd[label[2]] + hdd[label[3]])/3.0)

    def _trigger_epoch(self):
        if self.epoch_num > 0 and self.epoch_num % config.EVAL_EPOCH == 0:
            self._eval()

def offline_evaluate(pred_func, output_file):
        df = get_eval_dataflow()
        if config.DYNAMIC_SHAPE_PRED:    
            eval_brats(
                df, lambda img: segment_one_image_dynamic(img, pred_func))
        else:
            result = eval_brats(
                df, lambda img: segment_one_image(img, pred_func))
            print(str(result))

def offline_pred(pred_func, output_file):
    df = get_test_dataflow()
    pred_brats(
        df, lambda img: segment_one_image(img, pred_func))

def update_learning_rate_adam_exponential_decay(epoch, old_learning_rate):
    new_learning_rate = config.BASE_LR_ADAM * (1.0 - (epoch / config.MAX_EPOCH)) ** 0.9
    return new_learning_rate

def update_learning_rate_adam_with_warmup(epoch, old_learning_rate):
    if epoch < config.WARM_UP_MAX_EPOCH:
        new_learning_rate = config.BASE_LR_ADAM * (epoch / config.WARM_UP_MAX_EPOCH)
    else:
        new_learning_rate = config.BASE_LR_ADAM * (1.0 - (epoch - config.WARM_UP_MAX_EPOCH) / (config.MAX_EPOCH - config.WARM_UP_MAX_EPOCH)) ** 0.9

    return new_learning_rate
def original_scheduler(epoch):
    curr_epoch = epoch % 200
    ret = config.BASE_LR
    if curr_epoch < 40:
        ret = config.BASE_LR
    elif curr_epoch < 60:
        ret = config.BASE_LR * 0.1
    elif curr_epoch < 140:
        ret = config.BASE_LR * 0.01
    elif curr_epoch < 160:
        ret = config.BASE_LR * 0.001
    elif curr_epoch < 180:
        ret = config.BASE_LR * 0.0001
    elif curr_epoch < 200:
        ret = config.BASE_LR * 0.00001
    return ret

def low_learning_rate_scheduler(epoch):
    curr_epoch = epoch % 200
    ret = config.BASE_LR * config.LR_COEFF
    if curr_epoch < 40:
        ret = min(ret, config.BASE_LR)
    elif curr_epoch < 60:
        ret = min(ret, config.BASE_LR * 0.1)
    elif curr_epoch < 140:
        ret = min(ret, config.BASE_LR * 0.01)
    elif curr_epoch < 160:
        ret = min(ret, config.BASE_LR * 0.001)
    elif curr_epoch < 180:
        ret = min(ret, config.BASE_LR * 0.0001)
    elif curr_epoch < 200:
        ret = min(ret, config.BASE_LR * 0.00001)
    return ret

def train():
    logger.set_logger_dir(args.logdir)
    factor = get_batch_factor()
    stepnum = config.STEP_PER_EPOCH
    if args.adam:
        # lr = tf.get_variable('learning_rate', initializer=config.BASE_LR_ADAM, trainable=False)
        learning_rate_callback = HyperParamSetterWithFunc("learning_rate", lambda epoch, old_weight: update_learning_rate_adam_exponential_decay(epoch, old_weight))
    elif args.warmup_lr:
        # lr = tf.get_variable('learning_rate', initializer=config.BASE_LR_ADAM, trainable=False)
        learning_rate_callback = HyperParamSetterWithFunc("learning_rate", lambda epoch, old_weight: update_learning_rate_adam_with_warmup(epoch, old_weight))
    elif args.low_lr:
        learning_rate_callback = HyperParamSetterWithFunc('learning_rate', lambda epoch, old_weight: low_learning_rate_scheduler(epoch))
    else:
        # lr = tf.get_variable('learning_rate', initializer=config.BASE_LR, trainable=False)
        learning_rate_callback = HyperParamSetterWithFunc('learning_rate', lambda epoch, old_weight: original_scheduler(epoch))

    cfg = AutoResumeTrainConfig(
        model=get_model(),
        data=QueueInput(get_train_dataflow()),
        callbacks=[
            PeriodicCallback(
                ModelSaver(max_to_keep=5, keep_checkpoint_every_n_hours=1),
                every_k_epochs=config.EVAL_EPOCH),
            learning_rate_callback,
            # https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
            # If validation error wasn’t decreasing for 5 epochs, decay the learning rate by 0.2:
            # StatMonitorParamSetter('learning_rate', 'val_hdd',
            #             lambda x: x * 0.2, threshold=0, last_k=5),
            EvalCallback(),
            # MinSaver('val_hdd'),#, checkpoint_dir=args.logdir+'/ckpt_hdd/'), # tensorpack.utils.logger.get_logger_dir()
            MaxSaver('val_dice'),#, checkpoint_dir=args.logdir+'/ckpt_dice/'),
            #GPUUtilizationTracker(),
            #PeakMemoryTracker(),
            
            HyperParamSetterWithFunc("bd_loss_weight", lambda epoch, old_weight: config.BD_LOSS_LINEAR_WEIGHT * epoch if (epoch * config.BD_LOSS_LINEAR_WEIGHT < config.MAX_BD_FOCAL_LOSS_WEIGHT) else config.MAX_BD_FOCAL_LOSS_WEIGHT),
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

def predict():
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

def get_last_checkpoint():
    last_checkpoint = os.path.join(args.logdir, open(os.path.join(args.logdir, "checkpoint")).readline().rstrip()[24:-1])
    print('getting last checkpoint...', last_checkpoint)
    return last_checkpoint

def eval():
    if config.DYNAMIC_SHAPE_PRED:
        def get_dynamic_pred(shape):
            print(shape)
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
                session_init=get_model_loader(args.load if args.load else get_last_checkpoint()),
                input_names=['image'],
                output_names=get_model_output_names()))
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        #assert args.load
        offline_evaluate([pred], args.evaluate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to all availalbe ones', default='0')
    parser.add_argument('--load', help='load model for evaluation or training')
    parser.add_argument('--logdir', help='log directory', default='train-log-debug')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', action='store_true', help="Run evaluation")
    parser.add_argument('--predict', action='store_true', help="Run prediction")
    parser.add_argument('--fold', type=int, help="cross validation k fold")
    parser.add_argument('--multi_loss', action='store_true', help="multi-loss regression or not")
    parser.add_argument('--dataset', default=None, help="config dataset {brats2019 brats2020 iseg pancreas}")
    parser.add_argument('--debug', action='store_true', help="enter debug mode")
    parser.add_argument('--stair_case', action='store_true', help="stair case mode for C345")
    parser.add_argument('--rsu', action='store_true', help="RSU with CNN repace pooling")
    parser.add_argument('--transformerSA', type=int, default=1, help="transformer attention like for C12 and Spatial attention => expectedly enhance local features")
    parser.add_argument('--momentum', action='store_true', help="Momentum optimizer")
    parser.add_argument('--adam', action='store_true', help="Adam optimizer")
    parser.add_argument('--warmup_lr', action='store_true', help="warmup learning rate")
    parser.add_argument('--low_lr', type=float, default=1.0, help="trimmed learning rate")
    parser.add_argument('--no_f1_hdd', action='store_true', help="Not calculating f1 and hdd in eval")
    parser.add_argument('--boundary', action='store_true', help="Boundary loss")
    parser.add_argument('--boundary_focal', action="store_true", help="Boundary Focal loss")
    parser.add_argument('--focal_mode', default=0, type=int, help="Focal mode: 0 - normal, 1 - power, 2 - exp")
    parser.add_argument('--focal_coeff', default=-1, type=int, help="Focal Distance Map Coeff")
    parser.add_argument('--focal_func', default=0, type=int, help="Focal Function: 0 - identity, 1 - focal loss, 2 - tsa function, 3 - power function")
    parser.add_argument('--mixup_off', action='store_true', help="Turn off MixUp")

    # iseg
    parser.add_argument('--patch_size', type=int, help="config patch size")
    parser.add_argument('--step_size', type=int, help="step size")
    parser.add_argument('--step_per_epoch', type=int, default=500)
    parser.add_argument('--eval_epoch', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--gaussian_sigma_coeff', type=float, default=0, help="Gaussian sigma coefficient")
    args = parser.parse_args()
    if args.mixup_off:
        config.MIXUP = False
    config.EVALUATE = args.evaluate
    config.MULTI_LOSS = args.multi_loss
    config.stair_case = args.stair_case
    config.transformerSA = args.transformerSA
    config.RSU = args.rsu

    config.BOUNDARY_LOSS = args.boundary
    config.BOUNDARY_FOCAL= args.boundary_focal

    if config.BOUNDARY_LOSS == True:
        config.MIXUP = config.MIX_MATCH = False

    if config.BOUNDARY_FOCAL == True:
        config.MIXUP = config.MIX_MATCH = False
        config.FOCAL_MODE = args.focal_mode
        config.FOCAL_SDM_COEFF = args.focal_coeff
        config.PREDICTION_FOCAL_FUNCTION = args.focal_func


    if args.debug == True:
        args.dataset = 'brats2020'
        config.DEBUG = False
        config.NO_CACHE = False
        config.PATCH_SIZE = [64, 64, 64]
        config.INFERENCE_PATCH_SIZE = [64, 64, 64]
        input('confirm that the debug mode = ' + str(config.DEBUG) + ' no_cache mode = ' + str(config.NO_CACHE) + ' patch size config = ' + str(config.PATCH_SIZE))
        # config.STEP_PER_EPOCH = 500
        # config.EVAL_EPOCH = 80
        # config.MAX_EPOCH = 80
        config.STEP_PER_EPOCH = 200
        config.EVAL_EPOCH = 20
        config.MAX_EPOCH = 20
        config.MIX_MATCH = False ## fake
        args.fold = 0

    if args.patch_size is not None:
        config.PATCH_SIZE = [args.patch_size, args.patch_size, args.patch_size]
        config.INFERENCE_PATCH_SIZE = [args.patch_size, args.patch_size, args.patch_size]
    if args.step_size is not None:
        config.STEP_SIZE = args.step_size
    else:
        config.STEP_SIZE = config.INFERENCE_PATCH_SIZE[0]
    config.GAUSSIAN_SIGMA_COEFF = args.gaussian_sigma_coeff
    config.LR_COEFF = args.low_lr

    config.CROSS_VALIDATION = False # only use for Brats 2020
    if args.fold is not None:
        config.CROSS_VALIDATION = True
        config.FOLD = args.fold
        args.logdir = args.logdir + "_5fold-" + str(args.fold)

    if args.dataset is not None:
        config.DATASET = args.dataset
        if config.DATASET == 'iseg':
            config.idx2class = {0:'Background',1:'CSF',2:'GM',3:'WM'}
            config.STEP_PER_EPOCH = args.step_per_epoch
            config.EVAL_EPOCH = args.eval_epoch
            config.MAX_EPOCH = args.max_epoch
            config.save_pred = "save_eval/"+os.path.basename(args.logdir)
        elif args.dataset == 'pancreas':
            assert 1 == 2, "pls config your dataset folder"
        elif args.dataset == 'brats2018':
            config.BASEDIR = ["/media/Seagate16T/datasets/BraTS2018/"]
            config.TRAIN_DATASET = 'MICCAI_BraTS_2018_Data_Training'
            config.VAL_DATASET = 'MICCAI_BraTS_2018_Data_Validation'
            config.TEST_DATASET = None
            config.idx2class = {0:'Background',2:'WT',4:'TC',1:'ET'}
            assert 1 == 2, "test your dataset config for brats2018"
        elif args.dataset == 'brats2019':
            config.BASEDIR = ["/media/Seagate16T/datasets/BraTS2019/"]
            config.TRAIN_DATASET = 'MICCAI_BraTS_2019_Data_Training'
            config.VAL_DATASET = 'MICCAI_BraTS_2019_Data_Validation'
            config.TEST_DATASET = None
            config.idx2class = {0:'Background',2:'WT',4:'TC',1:'ET'}
        else:
            if args.dataset == 'brats2020':
                config.BASEDIR = ["/home/dghan/datasets/BraTS2020/"]
                config.TRAIN_DATASET = 'MICCAI_BraTS2020_TrainingData'
                config.VAL_DATASET = 'MICCAI_BraTS2020_ValidationData'
                config.TEST_DATASET = 'MICCAI_BraTS2020_TestingData'
                config.idx2class = {0:'Background',2:'WT',4:'TC',1:'ET'}
                if config.CROSS_VALIDATION == True:
                    config.BASEDIR = config.BASEDIR[0]
                    config.TRAIN_DATASET = 'training'
                    config.VAL_DATASET = 'val'
                    config.TEST_DATASET = 'val'
                    config.save_pred = '../save_pred/' + os.path.basename(args.logdir)
    else:
        assert 1 == 2, 'pls set the --dataset flag {brats2018 brats2019 brats2020 iseg pancreas}'

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if tf.test.gpu_device_name(): 
            print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        print(args.gpu)
        input('wait a second')

    print('=================================CONFIG===============================================')
    book = get_book_variable_module_name('config')
    for key, value in book.items():
        print(key,'\t',value)
    print('======================================================================================')
    sys.stdout = Logger()
    if args.visualize or args.evaluate:
        eval()    
    elif args.predict:
        predict()        
    else:
        train()

# python train.py --dataset brats2020 --logdir brats20_transformer6_loss3 --no_f1_hdd --fold 0 --transformerSA 6 --boundary_focal --focal_mode 1 --focal_coeff -1 --focal_func 3