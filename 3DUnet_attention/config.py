import numpy as np

# unet model
DATA_SAMPLING = 'one_positive' # one_positive, all_positive, random
###
# random: complete random sampling within entire volume.
# one_positive: at least one batch contain tumor label (label > 0)
# all_positive: all batch must contain tumor label (label > 0)
###
MIX_MATCH = False
MIXUP = True # False
RSU = False
RESIDUAL = True
DEPTH = 5
DEEP_SUPERVISION = True
FILTER_GROW = True
INSTANCE_NORM = True
# Use multi-view fusion 3 models for 3 view must be trained
DIRECTION = 'axial' # axial, sagittal, coronal
MULTI_VIEW = False

CA_attention = True
SA_attention = True

BASE_LR = 0.01

CROSS_VALIDATION = False
CROSS_VALIDATION_PATH = "./5folds.pkl"
FOLD = 0
###
# Use when 5 fold cross validation
# 1. First run generate_5fold.py to save 5fold.pkl
# 2. Set CROSS_VALIDATION to True
# 3. CROSS_VALIDATION_PATH to /path/to/5fold.pkl
# 4. Set FOLD to {0~4}
###
NO_CACHE = False
###
# if NO_CACHE = False, we load pre-processed volume into memory to accelerate training.
# set True when system memory loading is too high
###
TEST_FLIP = True
# Test time augmentation
DYNAMIC_SHAPE_PRED = False
# change PATCH_SIZE in inference if cropped brain region > PATCH_SIZE
ADVANCE_POSTPROCESSING = False
BATCH_SIZE = 1
PATCH_SIZE = [128, 128, 128]
INFERENCE_PATCH_SIZE = [128, 128, 128]
STEP_SIZE = 128
GAUSSIAN_SIGMA_COEFF = 0

INTENSITY_NORM = 'modality' # different norm method
STEP_PER_EPOCH = 500
EVAL_EPOCH = 30
MAX_EPOCH = 30#80
EVALUATE = False
DATASET=''
BASEDIR = ["/home/dghan/datasets/BraTS2019/"] #["/media/Seagate16T/datasets/BraTS2019/"]#"/home/dghan/datasets/BraTS2019/"]
TRAIN_DATASET = 'MICCAI_BraTS_2019_Data_Training'
VAL_DATASET = 'MICCAI_BraTS_2019_Data_Validation'
TEST_DATASET = 'MICCAI_BraTS2020_TestingData'
save_pred = "train-debug"

NUM_CLASS = 4
# GT_TEST = False

LR_COEFF = 1.0
BASE_LR_ADAM = 5e-4
WARM_UP_MAX_EPOCH = 10
BD_LOSS_LINEAR_WEIGHT = 0.3
MAX_BD_FOCAL_LOSS_WEIGHT = 0.3

MULTI_LOSS = False # True
DEBUG = False # enter debug mode, load less 2 subject only in train and 1 in callbackeval ~> default brats2020
stair_case = False
transformerSA = 1
idx2class = None

BOUNDARY_LOSS = False
# Distance map: focal
FOCAL_MODE_NORMAL   = 0
FOCAL_MODE_POWER    = 1
FOCAL_MODE_EXP      = 2
FOCAL_MODE_DILATE   = 3
FOCAL_MODE          = FOCAL_MODE_POWER

FOCAL_SDM_COEFF = -1

IDENTITY        = 0

FOCAL_FUNCTION  = 1
FOCAL_GAMMA     = 2 # from paper RetinaNet

# we borrow the exp function from paper Unsupervised Data Augmentation for Consistency Training
TSA_FUNCTION    = 2
TSA_MIN_LOSS_PARAM= 5 # The minimum loss is exp(- TSA_MIN_LOSS_PARAM)

POWER_FUNCTION  = 3
POWER_ALPHA     = 2

PREDICTION_FOCAL_FUNCTION = FOCAL_FUNCTION


###### INFERENCE #####
RESIZE = False
RESIZE_REF = False
FLOAT_INTERPOLATION = False
MASK_DILATION = 0
SLICE_STEP = 64

ET_vs_TC_THRESHOLD = 0.8
FLIP_LABEL_THRESHOLD = 1.6

SALIENCY_MAP = False
NUM_SAMPLE_SALIENCY_MAP = 1

# example usage:
# python train.py --gpu 2 --fold 0 --evaluate --load ./train_log/2020_5fold\ original/unet3d_2020_5fold-0/model-24500 | tee train_log/log2020-eval-no-gauss-fold0.txt