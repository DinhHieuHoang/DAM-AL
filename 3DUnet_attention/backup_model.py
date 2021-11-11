from keras.models import *
from attention import *
from bilinear_upsampling import BilinearUpsampling, BilinearUpsampling3D
import tensorflow as tf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope

from tensorpack.models import (
    layer_register
)
from custom_ops import BatchNorm3d, InstanceNorm5d
import numpy as np
# from scipy.ndimage import distance_transform_edt as distance
from scipy import ndimage
import scipy
import config
import tensorflow.contrib.slim as slim
import utils
import time
PADDING = "SAME"
DATA_FORMAT="channels_last"
BASE_FILTER = 16

class Copy(Layer):
    def call(self, inputs, **kwargs):
        copy = tf.identity(inputs)
        return copy
    def compute_output_shape(self, input_shape):
        return input_shape

def AtrousBlock3D(input_tensor, filters, rate, block_id, stride=1):
    x = tf.layers.conv3d(inputs=input_tensor, 
                   filters=filters,
                   kernel_size=(3,3,3),
                   strides=(stride, stride, stride),
                   dilation_rate=(rate, rate, rate),
                   padding=PADDING,
                   use_bias=False,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=block_id + "_dilation")
    # x = Conv3D(filters, (3, 3, 3), strides=(stride, stride, stride), dilation_rate=(rate, rate, rate),
    #            padding='same', use_bias=False, name=block_id + '_dilation')(input_tensor)
    return x

def CFE3D(input_tensor, filters, block_id):
    rate = [3, 5, 7]
    cfe0 = tf.layers.conv3d(inputs=input_tensor, 
                   filters=filters,
                   kernel_size=(1,1,1),
                   use_bias=False,
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=block_id + "_cfe0")
    # cfe0 = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False, name=block_id + '_cfe0')(
    #     input_tensor)
    cfe1 = AtrousBlock3D(input_tensor, filters, rate[0], block_id + '_cfe1')
    cfe2 = AtrousBlock3D(input_tensor, filters, rate[1], block_id + '_cfe2')
    cfe3 = AtrousBlock3D(input_tensor, filters, rate[2], block_id + '_cfe3')
    cfe_concat = tf.concat([cfe0, cfe1, cfe2, cfe3], axis=-1, name=block_id + 'concatcfe')
    #cfe_concat = Concatenate(name=block_id + 'concatcfe', axis=-1)([cfe0, cfe1, cfe2, cfe3])
    # with tf.variable_scope(block_id + "_BN") as scope:
    #     cfe_concat = BN_Relu(cfe_concat)
    return cfe_concat

@layer_register(log_shape=True)
def unet3d_attention(inputs):
    print("inputs ", inputs)
    depth = config.DEPTH
    filters = []
    down_list = []
    layer = tf.layers.conv3d(inputs=inputs, 
                   filters=BASE_FILTER,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="init_conv")
    print(layer.name, layer.shape[1:])

    # if config.RSU:
    #     mid_ch = BASE_FILTER // 2 # for first RSU mid channel
    #     print('RSU at C12 with pooling replaced by conv3d')
    for d in range(depth):
        if config.FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        # if config.RSU:
        #     # if depth < 7:
        #     #     height = 7 - d
        #     # else:
        #     #     height = depth - d

        #     height = 7 - d
        #     if height < 6:
        #         layer = Unet3dBlock('down{}'.format(d), layer, kernels=(3,3,3), n_feat=num_filters, s=1)
        #         print("Unet downsampling ",d,"    ",layer.shape[1:])
        #     else:
        #         layer = RSU('down{}RSU'.format(height), height, layer, mid_ch, num_filters)
        #         print("RSU ",height,"    ",layer.shape[1:])

        #     # if height < 4: # from stage 5 and more => change dilated into True
        #     #     layer = RSU('down{}RSU4'.format(d), 4, layer, mid_ch, num_filters, False)
        #     #     print("RSU4-"+str(d)+"    ",layer.shape[1:])
        #     # else:
        #     #     layer = RSU('down{}RSU'.format(d), height, layer, mid_ch, num_filters)
        #     #     print("RSU ",d,"    ",layer.shape[1:])

        #     mid_ch = num_filters // 2
        # else:
            layer = Unet3dBlock('down{}'.format(d), layer, kernels=(3,3,3), n_feat=num_filters, s=1)
            print("Residual bock downsampling ",d,"    ",layer.shape[1:])
        down_list.append(layer)
        if d != depth - 1:
            layer = tf.layers.conv3d(inputs=layer, 
                                    filters=num_filters*2,
                                    kernel_size=(3,3,3),
                                    strides=(2,2,2),
                                    padding=PADDING,
                                    activation=lambda x, name=None: BN_Relu(x),
                                    data_format=DATA_FORMAT,
                                    name="stride2conv{}".format(d))
            print("Down Conv3D ",d, "   ", layer.shape[1:])
        # print(layer.name,layer.shape[1:])

    C1 = tf.layers.conv3d(inputs=down_list[0], 
                   filters=64,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C1_conv")

    C2 = tf.layers.conv3d(inputs=down_list[1], 
                   filters=64,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C2_conv")

    print("Low level feature 1\t", C1.shape[1:])
    print("Low level feature 2\t", C2.shape[1:])

    C3_cfe = CFE3D(down_list[2], 32, 'C3_cfe')
    print("High level feature 1 CFE\t", C3_cfe.shape[1:])
    C4_cfe = CFE3D(down_list[3], 32, 'C4_cfe')
    print("High level feature 2 CFE\t", C4_cfe.shape[1:])
    C5_cfe = CFE3D(down_list[4], 32, 'C5_cfe')
    print("High level feature 3 CFE\t", C5_cfe.shape[1:])

    if config.stair_case:
        C5_cfe = UnetUpsample('C5_cfe_up4', C5_cfe, 2, 128)
        C345 = tf.concat([C4_cfe, C5_cfe], axis=-1, name='C45_concat')
        C345 = UnetUpsample('C45_up2', C4_cfe, 2, 128)
        C345 = tf.concat([C345, C3_cfe], axis=-1, name='C345_aspp_concat_stair_case')
        print("@Stair case version High level features aspp concat\t", C345.shape[1:])
    else:    
        C5_cfe = UnetUpsample('C5_cfe_up4', C5_cfe, 4, 128)
        C4_cfe = UnetUpsample('C4_cfe_up2', C4_cfe, 2, 128)
        C345 = tf.concat([C3_cfe, C4_cfe, C5_cfe], axis=-1, name='C345_aspp_concat')
        print("High level features aspp concat\t", C345.shape[1:])

    if config.CA_attention:
        C345 = ChannelWiseAttention3D(C345, name='C345_ChannelWiseAttention_withcpfe')
        print('High level features CA\t', C345.shape[1:])

    C345 = tf.layers.conv3d(inputs=C345, 
                   filters=64,
                   kernel_size=(1,1,1),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C345_conv")
    print('High level features conv\t', C345.shape[1:])
    C345 = UnetUpsample('C345_up4', C345, 4, 64)
    print('High level features upsampling\t', C345.shape[1:])

    if config.SA_attention:
        SA = SpatialAttention3D(C345, 'spatial_attention')
        print('High level features SA\t', SA.shape[1:])
    
    C2 = UnetUpsample('C2_up2', C2, 2, 64)
    C12 = tf.concat([C1, C2], axis=-1, name='C12_concat')
    C12 = tf.layers.conv3d(inputs=C12, 
                   filters=64,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C12_conv")
    print('Low level feature conv\t', C12.shape[1:])
    if config.MULTI_LOSS == True:
        C12_backup = tf.identity(C12)

    if config.transformerSA > 1:
        C12_backup = tf.identity(C12)
        print('transformer spatial attention level: ' + str(config.transformerSA))

    if config.SA_attention:
        C12 = tf.math.multiply(SA, C12, name='C12_atten_mutiply')
        for i in range(1, config.transformerSA):
            SA = SpatialAttention3D(C12, 'spatial_attention_'+str(i+1))
            C12 = tf.math.multiply(SA, C12_backup, name='C12_atten_mutiply_'+str(i+1))
    
    fea = tf.concat([C12, C345], axis=-1, name='fuse_concat')
    print('Low + High level feature\t', fea.shape[1:])
    layer = tf.layers.conv3d(fea, 
                            filters=config.NUM_CLASS,
                            kernel_size=(3,3,3),
                            padding="SAME",
                            activation=tf.identity,
                            data_format=DATA_FORMAT,
                            name="final")

    if DATA_FORMAT == 'channels_first':
        layer = tf.transpose(layer, [0, 2, 3, 4, 1]) # to-channel last
    print("final", layer.shape[1:]) # [3, num_class, d, h, w]

    if config.MULTI_LOSS == True:
        C12 = tf.layers.conv3d(C12_backup, 
                            filters=config.NUM_CLASS,
                            kernel_size=(3,3,3),
                            padding="SAME",
                            activation=tf.identity,
                            data_format=DATA_FORMAT,
                            name="C12_4")
        C345 = tf.layers.conv3d(C345, 
                    filters=config.NUM_CLASS,
                    kernel_size=(3,3,3),
                    padding="SAME",
                    activation=tf.identity,
                    data_format=DATA_FORMAT,
                    name="C345_4")
        print("final C12", C12.shape[1:])
        print("final C345", C345.shape[1:])
        return C12, C345, layer
    return layer

def Upsample3D(prefix, l, scale=2):
    return tf.keras.layers.UpSampling3D(size=(scale,scale,scale), data_format=DATA_FORMAT)(l)

def RSU(name, height, in_ch, mid_ch, out_ch, dilated=False):
    def REBNCONV(in_ch=3, out_ch=3, dilate=1):
        return tf.layers.conv3d(inputs=in_ch, 
                   filters=out_ch,
                   kernel_size=(3,3,3),
                   dilation_rate=(1*dilate, 1*dilate, 1*dilate),
                   padding=PADDING,
                   use_bias=False,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=None)
    
    down_list = []
    down_list.append(REBNCONV(in_ch, out_ch))

    for i in range(1, height):
        dilate = 1 if not dilated else 2 ** (i - 1)
        down_list.append(REBNCONV(down_list[i-1], mid_ch, dilate=dilate))
        down_list[i] = tf.layers.conv3d(inputs=down_list[i], filters=mid_ch, strides=(2,2,2), kernel_size=(2,2,2))#tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(down_list[i])

    for i in range(height, height+1):
        dilate = 1 if not dilated else 2 ** (i - 1)
        down_list.append(REBNCONV(down_list[i-1], mid_ch, dilate=dilate))

    up_layer = down_list[height]
    for i in range(height-1, 1, -1):
        dilate = 1 if not dilated else 2 ** (i - 1)
        up_layer = tf.concat([up_layer, down_list[i]], axis=-1)
        up_layer = REBNCONV(up_layer, mid_ch, dilate=dilate)
        up_layer = tf.keras.layers.UpSampling3D(size=(2,2,2), data_format=DATA_FORMAT)(up_layer)

    up_layer = tf.concat([up_layer, down_list[1]], axis=-1)
    up_layer = REBNCONV(up_layer, out_ch, dilate=dilate)
    up_layer = tf.keras.layers.UpSampling3D(size=(2,2,2), data_format=DATA_FORMAT)(up_layer)

    return up_layer + down_list[0]

def UnetUpsample(prefix, l, scale, num_filters):
    """
    l = tf.layers.conv3d_transpose(inputs=l, 
                                filters=num_filters,
                                kernel_size=(2,2,2),
                                strides=2,
                                padding=PADDING,
                                activation=tf.nn.relu,
                                data_format=DATA_FORMAT,
                                name="up_conv0_{}".format(prefix))
    """
    l = Upsample3D('', l, scale)
    l = tf.layers.conv3d(inputs=l, 
                        filters=num_filters,
                        kernel_size=(3,3,3),
                        strides=1,
                        padding=PADDING,
                        activation=lambda x, name=None: BN_Relu(x),
                        data_format=DATA_FORMAT,
                        name="up_conv1_{}".format(prefix))
    return l

def BN_Relu(x):
    if config.INSTANCE_NORM:
        l = InstanceNorm5d('ins_norm', x, data_format=DATA_FORMAT)
    else:
        l = BatchNorm3d('bn', x, axis=1 if DATA_FORMAT == 'channels_first' else -1)
    l = tf.nn.relu(l)
    return l

def Unet3dBlock(prefix, l, kernels, n_feat, s):
    if config.RESIDUAL:
        l_in = l

    for i in range(2):
        l = tf.layers.conv3d(inputs=l, 
                   filters=n_feat,
                   kernel_size=kernels,
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="{}_conv_{}".format(prefix, i))

    return l_in + l if config.RESIDUAL else l

### from niftynet ####
def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):

    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score


def dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """



    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)

    ids = tf.stack([ids, ground_truth], axis=1)

    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))


    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    
    return 1.0 - tf.reduce_mean(dice_score)

def dice_mixup(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.reduce_sum(
            weight_map_nclasses * ground_truth * prediction, axis=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.reduce_sum(tf.square(ground_truth) * weight_map_nclasses,
                                 axis=[0])
    else:
        dice_numerator = 2.0 * tf.reduce_sum(
            ground_truth * prediction, axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.reduce_sum(tf.square(ground_truth), axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    
    return 1.0 - tf.reduce_mean(dice_score)

def _cal_signed_distance_map(posmask):
    # given positive mask, calculate corresponding signed distance map 
    # return has the same shape with that of the input
    negmask = ~posmask
    posdis = scipy.ndimage.distance_transform_edt(posmask)
    negdis = scipy.ndimage.distance_transform_edt(negmask)
    res = negdis * np.array(negmask, dtype=np.float)
    res = res - (posdis - 1.0) * np.array(posmask, dtype=np.float)
    return res

def signed_distance_map(ground_truth):
    """
    Function re-written from https://github.com/JunMa11/SegWithDistMap. 
    Compute the signed distance map of the ground truth
        Paper: Kervadec et al., Boundary loss for highly unbalanced segmentation
    
    Parameters
    ----------
    ground_truth: array_like
        The segmentation ground truth, shape=(x,y,z), value: 0-background, 1-ET, 2-WT, 3-CT
    
    Returns
    -------
    ground_truth_sdm: array_like
        The signed distance map derived from the ground truth, shape=(x, y, z, label)
    """

    res = None
    for idx in range(1, config.NUM_CLASS):
        posmask = ground_truth == idx
        sdm = None
        if posmask.any():
            sdm = _cal_signed_distance_map(posmask)
        else:
            sdm = np.ones(posmask.shape)

        if idx == 1:
            res = np.array([sdm])
        else:
            res = np.concatenate((res, [sdm]), axis=0)

    return res

def _get_sdm(ground_truth, idx): # 1: ET   2: WT   3: TC
    posmask = ground_truth == idx
    if posmask.any():
        sdm = _cal_signed_distance_map(posmask)
    else:
        sdm = np.ones(posmask.shape)
    return sdm # numpy ndarray

def modified_distance_map(ground_truth, mode=config.FOCAL_MODE, coeff=config.FOCAL_SDM_COEFF): 
    """
    Returns new processed distance map
    """
    res = None
    if mode == config.FOCAL_MODE_POWER: 
        """
        TODO write docs
        """
        def power_dm(sdm, coeff=coeff):
            dm = np.abs(sdm)
            if coeff < 0 and (dm == 0).any(): # power with negative number -> must add ones to make all zeros greater than zeros
                dm = dm + np.ones(sdm.shape)
            dm = np.power(dm, coeff)
            return dm
        
        for idx in range(1, config.NUM_CLASS):
            dm = power_dm(_get_sdm(ground_truth, idx))
            res = np.array([dm]) if idx == 1 else np.concatenate((res, [dm]), axis=0)

    elif mode == config.FOCAL_MODE_EXP: # 
        """
        TODO write docs
        """
        def exp_sdm(sdm):
            dm = -1 * np.abs(sdm)
            dm = np.exp(dm)
            return dm

        for idx in range(1, config.NUM_CLASS):
            dm = exp_sdm(_get_sdm(ground_truth, idx))
            res = np.array([dm]) if idx == 1 else np.concatenate((res, [dm]), axis=0)

    elif mode == config.FOCAL_MODE_DILATE:
        for idx in range(1, config.NUM_CLASS):
            idx_map = np.array(ground_truth) == idx
            struct  = ndimage.generate_binary_structure(3, 1)
            dilation_map = ndimage.binary_dilation(input=idx_map, structure=struct, iterations=coeff)
            erosion_map  = ndimage.binary_erosion(input=idx_map, structure=struct, iterations=coeff)
            sdm = np.logical_xor(dilation_map, erosion_map).astype(np.float32)
            sdm[idx == True] = -1
            res = np.array([sdm]) if idx == 1 else np.concatenate((res, [sdm]), axis=0)
    else: # default: signed distance map
        res = signed_distance_map_with_edt(ground_truth)
    return res

def prediction_focal(prediction, ground_truth, mode=config.IDENTITY):
    """
    Note that the background prediction will be discarded
    """
    def get_sign_map(idx):
        sign_map = tf.where(tf.equal(ground_truth, tf.constant(idx, dtype=tf.float32)), tf.ones_like(ground_truth) * -1.0, tf.ones_like(ground_truth)) # binary 0-1
        return sign_map
    
    def get_pred_map(idx):
        pred_map = prediction[idx]
        return pred_map

    proc_pred = np.array([])
    if mode == config.FOCAL_FUNCTION:
        # discard background (idx=0) and process remains
        for idx in range(1, config.NUM_CLASS):
            sign = get_sign_map(idx)
            pred = get_pred_map(idx)
            pred = tf.where(tf.equal(sign, 1.0), 1.0 - pred, pred)
            # focal loss - -(1 - x)^gamma * log(x)
            pred = tf.pow((1.0 - pred), config.FOCAL_GAMMA) * tf.log(pred) * (-1)
            proc_pred = tf.expand_dims(pred, axis=0) if idx == 1 else tf.concat((proc_pred, tf.expand_dims(pred, axis=0)), axis=0)

    elif mode == config.TSA_FUNCTION:
        # discard background (idx=0) and process remains
        for idx in range(1, config.NUM_CLASS):
            sign = get_sign_map(idx)
            pred = get_pred_map(idx)
            pred = tf.where(tf.equal(sign, 1.0), 1.0 - pred, pred)
            # Training Signal Annealing func - exp_schedule - e^(-5x)
            pred = tf.exp(-1.0 * pred * config.TSA_MIN_LOSS_PARAM)
            proc_pred = tf.expand_dims(pred, axis=0) if idx == 1 else tf.concat((proc_pred, tf.expand_dims(pred, axis=0)), axis=0)

    elif mode == config.POWER_FUNCTION:
        for idx in range(1, config.NUM_CLASS):
            sign = get_sign_map(idx)
            pred = get_pred_map(idx)
            pred = tf.where(tf.equal(sign, 1.0), 1.0 - pred, pred)
            # Power func - (1-x)^alpha
            pred = tf.pow((1.0 - pred), config.POWER_ALPHA)
            proc_pred = tf.expand_dims(pred, axis=0) if idx == 1 else tf.concat((proc_pred, tf.expand_dims(pred, axis=0)), axis=0)

    else: # config.IDENTITY or anything else
        for idx in range(1, config.NUM_CLASS):
            pred = get_sign_map(idx) * get_pred_map(idx)
            proc_pred = tf.expand_dims(pred, axis=0) if idx == 1 else tf.concat((proc_pred, tf.expand_dims(pred, axis=0)), axis=0)
    return proc_pred

def boundary_focal_loss(prediction, ground_truth, ground_truth_dm):
    """
    New loss which is the combination of the parameterized boundary loss and Focal Loss idea

    Parameters
    ----------
    prediction: array_like
        The logits, shape=(x, y, z, label)
    ground_truth: array_like
        The segmentation ground truth, shape=(x,y,z), value: 0-background, 1-ET, 2-WT, 3-CT

    Returns
    -------
    bd_focal_loss: float
    """
    # process prediction
    prediction = tf.cast(prediction, tf.float32)
    prediction = tf.transpose(prediction, perm=(3, 0, 1, 2)) # transpose to (n_class, depth, height, width)
    ground_truth_dm = tf.transpose(ground_truth_dm, perm=(3,0,1,2))
    print("boundary_focal_loss", prediction.get_shape(), ground_truth.get_shape())
    # prediction = tf.py_function(func=prediction_focal, inp=[prediction, ground_truth, config.PREDICTION_FOCAL_FUNCTION], Tout=tf.float32)
    prediction = prediction_focal(prediction, ground_truth, config.PREDICTION_FOCAL_FUNCTION)
    print("boundary_focal_loss", prediction.get_shape(), ground_truth_dm.get_shape())
    # combine
    weight_sum = tf.reduce_sum(ground_truth_dm)
    multiplied = tf.einsum("cxyz, cxyz -> cxyz", prediction, ground_truth_dm)
    # bd_focal_loss = tf.reduce_mean(multiplied) # make the gradient too small
    bd_focal_sum = tf.reduce_sum(multiplied)
    bd_focal_loss= tf.divide(bd_focal_sum, weight_sum)
    return bd_focal_loss

def boundary_loss(prediction, ground_truth_dm):
    """
    Function re-written from https://github.com/JunMa11/SegWithDistMap. 
    Compute the signed distance map of the ground truth.
        Paper: Kervadec et al., Boundary loss for highly unbalanced segmentation

    Parameters
    ----------
    prediction: array_like
        The logits, shape=(x, y, z, label)
    ground_truth: array_like
        The segmentation ground truth, shape=(x,y,z), value: 0-background, 1-ET, 2-WT, 3-CT

    Returns
    -------
    boundary_loss: float
    """
    # process prediction
    prediction = tf.cast(prediction, tf.float32)
    prediction = tf.transpose(prediction, perm=(3, 0, 1, 2)) # transpose to (n_class, depth, height, width)
    ground_truth_dm = tf.transpose(ground_truth_dm, perm=(3,0,1,2))
    sliced_prediction = prediction[1:] # discard background layer?
    # combine
    multiplied = tf.einsum("cxyz, cxyz -> cxyz", sliced_prediction, ground_truth_dm) # [n_class except background, depth, height, width]
    bd_loss = tf.reduce_mean(multiplied)
    return bd_loss

def Loss(feature, weight, gt, distance_map=None):
    # compute batch-wise
    losses = []
    bd_loss_weight = tf.get_variable("bd_loss_weight", initializer=config.BD_LOSS_LINEAR_WEIGHT, trainable=False)
    dc_loss_weight = 1.0 - bd_loss_weight
    bd_losses=[]
    dc_losses=[]
    for idx in range(config.BATCH_SIZE):
        f = tf.reshape(feature[idx], [-1, config.NUM_CLASS]) # Flatten feature into [|volume|, 4 - num_class]
        #f = tf.cast(f, dtype=tf.float32)
        #f = tf.nn.softmax(f)
        w = tf.reshape(weight[idx], [-1]) # Flatten into 1D array
        if config.MIXUP:
            g = tf.reshape(gt[idx], [-1, config.NUM_CLASS]) # Flatten ground truth into [|volume|, 4 - num_class]
        else:
            g = tf.reshape(gt[idx], [-1]) # Flatten into 1D array
        if g.shape.as_list()[-1] == 1:
            g = tf.squeeze(g, axis=-1) # (nvoxel, )
        if w.shape.as_list()[-1] == 1:
            w = tf.squeeze(w, axis=-1) # (nvoxel, )
        f = tf.nn.softmax(f)
        if config.MIXUP:
            loss_per_batch = dice_mixup(f, g, weight_map=w)
        else: # MIXUP == False
            if config.BOUNDARY_LOSS:
                bd_loss = boundary_loss(tf.nn.softmax(feature[idx]), distance_map[idx])
                dc_loss = dice(f, g, weight_map=w)

                bd_losses.append(bd_loss)
                dc_losses.append(dc_loss)

                loss_per_batch = (dc_loss * dc_loss_weight) + (bd_loss * bd_loss_weight)
            elif config.BOUNDARY_FOCAL:
                if gt[idx].shape.as_list()[-1] == 1:
                    ground_truth = tf.squeeze(gt[idx], axis=-1)
                bd_focal_loss = boundary_focal_loss(tf.nn.softmax(feature[idx]), ground_truth, distance_map[idx])
                dc_loss = dice(f, g, weight_map=w)
        
                bd_losses.append(bd_focal_loss)
                dc_losses.append(dc_loss)

                loss_per_batch = (dc_loss * dc_loss_weight) + (bd_focal_loss * bd_loss_weight)
            else:
                loss_per_batch = dice(f, g, weight_map=w)
            
        # loss_per_batch = cross_entropy(f, g, weight_map=w)
        losses.append(loss_per_batch)
    if config.BOUNDARY_LOSS:
        tf.summary.scalar('bd_loss_weight', bd_loss_weight)
        tf.summary.scalar('bd_loss', tf.reduce_mean(bd_losses))
        tf.summary.scalar('dc_loss_weight', dc_loss_weight)
        tf.summary.scalar('dc_loss', tf.reduce_mean(dc_losses))
        tf.summary.scalar('loss', tf.reduce_mean(losses))
    elif config.BOUNDARY_FOCAL:
        tf.summary.scalar('bd_loss_weight', bd_loss_weight)
        tf.summary.scalar('bd_focal_loss', tf.reduce_mean(bd_losses))
        tf.summary.scalar('dc_loss_weight', dc_loss_weight)
        tf.summary.scalar('dc_loss', tf.reduce_mean(dc_losses))
        tf.summary.scalar('loss', tf.reduce_mean(losses))
    return tf.reduce_mean(losses, name="dice_loss")

