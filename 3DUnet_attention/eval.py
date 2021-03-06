# # -*- coding: utf-8 -*-
# # File: eval.py
# from __future__ import print_function
# import tqdm
# import os
# from collections import namedtuple
# import numpy as np
# #import cv2
# import time

# from tensorpack.utils.utils import get_tqdm_kwargs
# import config
# from utils import *
# import nibabel as nib
# from scipy.special import softmax
# from MedPy import calculate_f1_hdd
# from scipy import signal

# def get_gaussian_kernel(size, sigma = None):
#     # size 128 => np.arange(-63,65,1);      size 7 => np.arange(-3,4,1)
#     if sigma is None: sigma = size*2.0/3.0
#     mid = size // 2 + 1
#     premid = mid - size 
#     x = np.arange(premid,mid,1)   # coordinate arrays -- make sure they contain 0!
#     y = np.arange(premid,mid,1)
#     z = np.arange(premid,mid,1)
#     xx, yy, zz = np.meshgrid(x,y,z)
#     kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
#     return kernel/np.max(kernel[:, mid, mid])

# def post_processing(pred1, temp_weight):
#     struct = ndimage.generate_binary_structure(3, 2)
#     margin = 5
#     wt_threshold = 2000
#     pred1 = pred1 * temp_weight # clear non-brain region
#     # pred1 should be the same as cropped brain region
#     # now fill the croped region with our prediction
#     pred_whole = np.zeros_like(pred1)
#     pred_core = np.zeros_like(pred1)
#     pred_enhancing = np.zeros_like(pred1)
#     pred_whole[pred1 > 0] = 1
#     pred1[pred1 == 2] = 0
#     pred_core[pred1 > 0] = 1
#     pred_enhancing[pred1 == 4]  = 1
    
#     pred_whole = ndimage.morphology.binary_closing(pred_whole, structure = struct)
#     pred_whole = get_largest_two_component(pred_whole, False, wt_threshold)
    
#     sub_weight = np.zeros_like(temp_weight)
#     sub_weight[pred_whole > 0] = 1
#     pred_core = pred_core * sub_weight
#     pred_core = ndimage.morphology.binary_closing(pred_core, structure = struct)
#     pred_core = get_largest_two_component(pred_core, False, wt_threshold)

#     subsub_weight = np.zeros_like(temp_weight)
#     subsub_weight[pred_core > 0] = 1
#     pred_enhancing = pred_enhancing * subsub_weight
#     vox_3  = np.asarray(pred_enhancing > 0, np.float32).sum()
#     all_vox = np.asarray(pred_whole > 0, np.float32).sum()
#     if(all_vox > 100 and 0 < vox_3 and vox_3 < 100):
#         pred_enhancing = np.zeros_like(pred_enhancing)
#     out_label = pred_whole * 2
#     out_label[pred_core>0] = 1
#     out_label[pred_enhancing>0] = 4

#     return out_label

# def batch_segmentation(temp_imgs, model_func, data_shape=[19, 180, 160]):
#     batch_size = config.BATCH_SIZE
#     data_channel = 4
#     class_num = config.NUM_CLASS
#     image_shape = temp_imgs[0].shape
#     label_shape = [data_shape[0], data_shape[1], data_shape[2]]
#     D, H, W = image_shape
#     input_center = [int(D/2), int(H/2), int(W/2)]
#     temp_prob1 = np.zeros([D, H, W, class_num])

#     sub_image_batches = []
#     for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):
#         center_slice = min(center_slice, D - int(label_shape[0]/2))
#         sub_image_batch = []
#         for chn in range(data_channel):
#             temp_input_center = [center_slice, input_center[1], input_center[2]]
#             sub_image = extract_roi_from_volume(
#                             temp_imgs[chn], temp_input_center, data_shape, fill="zero")
#             sub_image_batch.append(sub_image)
#         sub_image_batch = np.asanyarray(sub_image_batch, np.float32) #[4,180,160]
#         sub_image_batches.append(sub_image_batch) # [14,4,d,h,w]
    
#     total_batch = len(sub_image_batches)
#     max_mini_batch = int((total_batch+batch_size-1)/batch_size)
#     sub_label_idx1 = 0
#     for mini_batch_idx in range(max_mini_batch):
#         data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
#                                       min((mini_batch_idx+1)*batch_size, total_batch)]
#         if(mini_batch_idx == max_mini_batch - 1):
#             for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
#                 data_mini_batch.append(np.zeros([data_channel] + data_shape))
#                 # data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
#         data_mini_batch = np.asarray(data_mini_batch, np.float32)
#         data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
#         prob_mini_batch1, _ = model_func(data_mini_batch)
        
#         for batch_idx in range(prob_mini_batch1.shape[0]):
#             center_slice = sub_label_idx1*label_shape[0] + int(label_shape[0]/2)
#             center_slice = min(center_slice, D - int(label_shape[0]/2))
#             temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]
#             sub_prob = np.reshape(prob_mini_batch1[batch_idx], label_shape + [class_num])
#             temp_prob1 = set_roi_to_volume(temp_prob1, temp_input_center, sub_prob)
#             sub_label_idx1 = sub_label_idx1 + 1
    
#     return temp_prob1

# def overlapping_inference(temp_imgs, model_func, data_shape):
#     start = time.time()
#     crop_size = data_shape
#     xstep = ystep = zstep = config.INFERENCE_PATCH_SIZE[0]# 128 # 16 #64 #@dghan

#     image = temp_imgs
#     image = np.array(image)
#     image = np.rollaxis(image, 0, 4)
#     image = np.expand_dims(image, 0)
#     #print(image.shape)

#     _, D, H, W, _ = image.shape
#     deep_slices   = np.arange(0, max(1, D - crop_size[0] + xstep), xstep)
#     height_slices = np.arange(0, max(1, H - crop_size[1] + ystep), ystep)
#     width_slices  = np.arange(0, max(1, W - crop_size[2] + zstep), zstep)

#     whole_pred = np.zeros(image.shape[:-1] + (config.NUM_CLASS,))
#     #print(whole_pred.shape)
#     count_used = np.zeros((D, H, W))

#     gaussian_kernel = get_gaussian_kernel(config.INFERENCE_PATCH_SIZE[0], config.INFERENCE_PATCH_SIZE[0]*1.5)

#     for j in range(len(deep_slices)):
#         for k in range(len(height_slices)):
#             for l in range(len(width_slices)):
#                 deep = deep_slices[j]
#                 height = height_slices[k]
#                 width = width_slices[l]
#                 image_input = np.zeros(shape = (config.BATCH_SIZE,) + tuple(data_shape) + (4 if config.DATASET != 'iseg' else 2,))
#                 # image_input = np.zeros(shape = (config.BATCH_SIZE,) + tuple(data_shape) + (4,))
#                 image_crop = image[:, deep   : deep   + crop_size[0],
#                                     height : height + crop_size[1],
#                                     width  : width  + crop_size[2], :]
#                 image_input[:, :image_crop.shape[1], :image_crop.shape[2], :image_crop.shape[3], :] = image_crop

#                 pred, _ = model_func(image_input)
#                 # pred[0,:,:,:,0] = pred[0,:,:,:,0]*gaussian_kernel
#                 # pred[0,:,:,:,1] = pred[0,:,:,:,1]*gaussian_kernel
#                 # pred[0,:,:,:,2] = pred[0,:,:,:,2]*gaussian_kernel
#                 # pred[0,:,:,:,3] = pred[0,:,:,:,3]*gaussian_kernel
#                 #print(outputs[0].shape)
#                 #----------------Average-------------------------------
#                 whole_pred[:, deep: deep + crop_size[0],
#                             height: height + crop_size[1],
#                             width: width + crop_size[2], :] += pred[:, :image_crop.shape[1], :image_crop.shape[2], :image_crop.shape[3], :]

#                 count_used[deep: deep + crop_size[0],
#                             height: height + crop_size[1],
#                             width: width + crop_size[2]] += 1

#     count_used = np.expand_dims(count_used, (0, -1))
#     whole_pred = whole_pred / count_used

#     return np.squeeze(whole_pred)

# def segment_one_image(data, model_func, is_online=False):
#     """
#     perform inference and unpad the volume to original shape
#     """
#     img = data['images']
#     temp_weight = data['weights'][:,:,:,0]
#     temp_size = data['original_shape']
#     temp_bbox = data['bbox']
#     # Ensure online evaluation match the training patch shape...should change in future 
#     batch_data_shape = config.PATCH_SIZE if is_online else config.INFERENCE_PATCH_SIZE
    
#     img = img[np.newaxis, ...] # add batch dim

#     im = img

#     if config.MULTI_VIEW:
#         im_ax = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_ax = transpose_volumes(im_ax, 'axial')
#         prob1_ax = batch_segmentation(im_ax, model_func[0], data_shape=batch_data_shape)

#         im_sa = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_sa = transpose_volumes(im_sa, 'sagittal')
#         prob1_sa = batch_segmentation(im_sa, model_func[1], data_shape=batch_data_shape)

#         im_co = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_co = transpose_volumes(im_co, 'coronal')
#         prob1_co = batch_segmentation(im_co, model_func[2], data_shape=batch_data_shape)

#         prob1 = (prob1_ax + np.transpose(prob1_sa, (1, 2, 0, 3)) + np.transpose(prob1_co, (1, 0, 2, 3))) / 3.0
#         pred1 = np.argmax(prob1, axis=-1)
        
#     else:
#         im_pred = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_pred = transpose_volumes(im_pred, config.DIRECTION)
#         # prob1 = batch_segmentation(im_pred, model_func[0], data_shape=batch_data_shape)
#         prob1 = overlapping_inference(im_pred, model_func[0], data_shape=batch_data_shape)
#         if config.DIRECTION == 'sagittal':
#             prob1 = np.transpose(prob1, (1, 2, 0, 3))
#         elif config.DIRECTION == 'coronal':
#             prob1 = np.transpose(prob1, (1, 0, 2, 3))
#         else:
#             prob1 = prob1
        
#         if config.NUM_CLASS == 1:
#             pred1 = prob1 >= 0.5
#             pred1 = np.squeeze(pred1, axis=-1)
#         else:
#             pred1 = np.argmax(prob1, axis=-1)
    
#     pred1[pred1 == 3] = 4
#     # pred1 should be the same as cropped brain region
#     if config.ADVANCE_POSTPROCESSING:
#         out_label = post_processing(pred1, temp_weight)
#     else:
#         out_label = pred1
#     out_label = np.asarray(out_label, np.int16)

#     if 'is_flipped' in data and data['is_flipped']:
#         out_label = np.flip(out_label, axis=-1)
#         prob1 = np.flip(prob1, axis=2) # d, h, w, num_class
    
#     final_label = np.zeros(temp_size, np.int16)
#     final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)

#     final_probs = np.zeros(list(temp_size) + [config.NUM_CLASS], np.float32)
#     final_probs = set_ND_volume_roi_with_bounding_box_range(final_probs, temp_bbox[0]+[0], temp_bbox[1]+[config.NUM_CLASS - 1], prob1)
        
#     return final_label, final_probs

# def dice_of_brats_data_set(gt, pred, type_idx):
#     dice_all_data = []
#     for i in range(len(gt)):
#         g_volume = gt[i]
#         s_volume = pred[i]
#         dice_one_volume = []
#         if(type_idx ==0): # whole tumor
#             if config.NUM_CLASS == 2:
#                 g_volume[g_volume == 4] = 1
#                 g_volume[g_volume == 2] = 1
#             temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
#             dice_one_volume = [temp_dice]
#         elif(type_idx == 1): # tumor core
#             s_volume[s_volume == 2] = 0
#             g_volume[g_volume == 2] = 0
#             temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
#             dice_one_volume = [temp_dice]
#         else:
#             #for label in [1, 2, 3, 4]: # dice of each class
#             temp_dice = binary_dice3d(s_volume == 4, g_volume == 4)
#             dice_one_volume = [temp_dice]
#         dice_all_data.append(dice_one_volume)
#     return dice_all_data

# def eval_brats(df, detect_func, with_gt=True, save_nii=True, no_f1_hdd=False):
#     """
#     evalutation
#     """
#     df.reset_state()
#     gts = []
#     results = []
#     gts_filename = []
#     i = 0
#     with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
#         for filename, image_id, data in df.get_data():
#             final_label, probs = detect_func(data)
#             if config.TEST_FLIP:
#                 pred_flip, probs_flip = detect_func(flip_lr(data))
#                 final_prob = (probs + probs_flip) / 2.0
#                 pred = np.argmax(final_prob, axis=-1)
#                 pred[pred == 3] = 4
#                 if config.ADVANCE_POSTPROCESSING:
#                     pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
#                     pred = post_processing(pred, data['weights'][:,:,:,0])
#                     pred = np.asarray(pred, np.int16)
#                     final_label = np.zeros(data['original_shape'], np.int16)
#                     final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
#                 else:
#                     final_label = pred
#             if save_nii:
#                 save_to_nii(final_label, image_id, outdir=config.save_pred, mode="label")
#             gt = load_nifty_volume_as_array("{}/{}_seg.nii.gz".format(filename, image_id))
#             gts_filename.append("{}/{}_seg.nii.gz".format(filename, image_id))
#             gts.append(gt)
#             results.append(final_label)
#             pbar.update()
#             i = i + 1
#             if config.DEBUG and i == 3:
#                 break
#     test_types = ['WT', 'TC', 'ET']
#     dices = {}
#     class_num = config.NUM_CLASS if config.NUM_CLASS == 1 else config.NUM_CLASS - 1
#     for type_idx in range(class_num):
#         dice = dice_of_brats_data_set(gts, results, type_idx)
#         dice = np.asarray(dice)
#         dice_mean = dice.mean(axis = 0)
#         dice_std  = dice.std(axis = 0)
#         test_type = test_types[type_idx]
#         dices[test_type] = dice_mean[0]
#     if no_f1_hdd:
#         return dices, None, None
#     f1, hdd = calculate_f1_hdd(gts, results, gts_filename)
#     return dices, f1, hdd

# def pred_brats(df, detect_func):
#     df.reset_state()
#     results = []

#     with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
#         for filename, image_id, data in df.get_data():
#             final_label, probs = detect_func(data)
#             if config.TEST_FLIP:
#                 pred_flip, probs_flip = detect_func(flip_lr(data))
#                 final_prob = (probs + probs_flip) / 2.0

#                 pred = np.argmax(final_prob, axis=-1)
#                 pred[pred == 3] = 4
#                 if config.ADVANCE_POSTPROCESSING:
#                     pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
#                     pred = post_processing(pred, data['weights'][:,:,:,0])
#                     pred = np.asarray(pred, np.int16)
#                     final_label = np.zeros(data['original_shape'], np.int16)
#                     final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
#                 else:
#                     final_label = pred
#             save_to_nii(final_label, image_id, outdir=config.save_pred, mode="label")
            
#             pbar.update()
#     return None

# iseg
# -*- coding: utf-8 -*-
# File: eval.py
from __future__ import print_function
import tqdm
import os
from collections import namedtuple
import numpy as np
import time

from tensorpack.utils.utils import get_tqdm_kwargs
import config
from utils import *
import nibabel as nib
from scipy.special import softmax
from MedPy import calculate_f1_hdd, calculate_dice_hdd_asd
from scipy import signal
import copy

def get_gaussian_kernel(size, sigma = None):
    # size 128 => np.arange(-63,65,1);      size 7 => np.arange(-3,4,1)
    if sigma is None: sigma = size*2.0/3.0
    mid = size // 2 + 1
    premid = mid - size 
    x = np.arange(premid,mid,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(premid,mid,1)
    z = np.arange(premid,mid,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    return kernel/np.max(kernel[:, mid, mid])

def post_processing(pred1, temp_weight):
    struct = ndimage.generate_binary_structure(3, 2)
    margin = 5
    wt_threshold = 2000
    pred1 = pred1 * temp_weight # clear non-brain region
    # pred1 should be the same as cropped brain region
    # now fill the croped region with our prediction
    pred_whole = np.zeros_like(pred1)
    pred_core = np.zeros_like(pred1)
    pred_enhancing = np.zeros_like(pred1)
    pred_whole[pred1 > 0] = 1
    pred1[pred1 == 2] = 0
    pred_core[pred1 > 0] = 1
    pred_enhancing[pred1 == 4]  = 1
    
    pred_whole = ndimage.morphology.binary_closing(pred_whole, structure = struct)
    pred_whole = get_largest_two_component(pred_whole, False, wt_threshold)
    
    sub_weight = np.zeros_like(temp_weight)
    sub_weight[pred_whole > 0] = 1
    pred_core = pred_core * sub_weight
    pred_core = ndimage.morphology.binary_closing(pred_core, structure = struct)
    pred_core = get_largest_two_component(pred_core, False, wt_threshold)

    subsub_weight = np.zeros_like(temp_weight)
    subsub_weight[pred_core > 0] = 1
    pred_enhancing = pred_enhancing * subsub_weight
    vox_3  = np.asarray(pred_enhancing > 0, np.float32).sum()
    all_vox = np.asarray(pred_whole > 0, np.float32).sum()
    if(all_vox > 100 and 0 < vox_3 and vox_3 < 100):
        pred_enhancing = np.zeros_like(pred_enhancing)
    out_label = pred_whole * 2
    out_label[pred_core>0] = 1
    out_label[pred_enhancing>0] = 4

    # out_label = change_brats_iseg_label(out_label, [2, 4, 1], [1, 2, 3]) ## iseg k cos loofng nhw trwsng gaf choox white matter vaf grey matter
    return out_label

def batch_segmentation(temp_imgs, model_func, data_shape=[19, 180, 160]):
    batch_size = config.BATCH_SIZE
    data_channel = 4
    class_num = config.NUM_CLASS
    image_shape = temp_imgs[0].shape
    label_shape = [data_shape[0], data_shape[1], data_shape[2]]
    D, H, W = image_shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob1 = np.zeros([D, H, W, class_num])

    sub_image_batches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):
        center_slice = min(center_slice, D - int(label_shape[0]/2))
        sub_image_batch = []
        for chn in range(data_channel):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            sub_image = extract_roi_from_volume(
                            temp_imgs[chn], temp_input_center, data_shape, fill="zero")
            sub_image_batch.append(sub_image)
        sub_image_batch = np.asanyarray(sub_image_batch, np.float32) #[4,180,160]
        sub_image_batches.append(sub_image_batch) # [14,4,d,h,w]
    
    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx1 = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.zeros([data_channel] + data_shape))
                # data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch1, _ = model_func(data_mini_batch)
        
        for batch_idx in range(prob_mini_batch1.shape[0]):
            center_slice = sub_label_idx1*label_shape[0] + int(label_shape[0]/2)
            center_slice = min(center_slice, D - int(label_shape[0]/2))
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]
            sub_prob = np.reshape(prob_mini_batch1[batch_idx], label_shape + [class_num])
            temp_prob1 = set_roi_to_volume(temp_prob1, temp_input_center, sub_prob)
            sub_label_idx1 = sub_label_idx1 + 1
    
    return temp_prob1

def overlapping_inference(temp_imgs, model_func, data_shape):
    start = time.time()
    crop_size = data_shape
    xstep = ystep = zstep = config.STEP_SIZE# config.INFERENCE_PATCH_SIZE[0] #8 # 16 #64 #@dghan

    image = temp_imgs
    image = np.array(image)
    image = np.rollaxis(image, 0, 4)
    image = np.expand_dims(image, 0)
    #print(image.shape)

    _, D, H, W, _ = image.shape
    deep_slices   = np.arange(0, max(1, D - crop_size[0] + xstep), xstep)
    height_slices = np.arange(0, max(1, H - crop_size[1] + ystep), ystep)
    width_slices  = np.arange(0, max(1, W - crop_size[2] + zstep), zstep)

    whole_pred = np.zeros(image.shape[:-1] + (config.NUM_CLASS,))
    #print(whole_pred.shape)
    count_used = np.zeros((D, H, W))

    if config.GAUSSIAN_SIGMA_COEFF > 0:
        gaussian_kernel = get_gaussian_kernel(config.INFERENCE_PATCH_SIZE[0], config.INFERENCE_PATCH_SIZE[0] * config.GAUSSIAN_SIGMA_COEFF)

    for j in range(len(deep_slices)):
        for k in range(len(height_slices)):
            for l in range(len(width_slices)):
                deep = deep_slices[j]
                height = height_slices[k]
                width = width_slices[l]
                image_input = np.zeros(shape = (config.BATCH_SIZE,) + tuple(data_shape) + (4 if config.DATASET != 'iseg' else 2,))
                image_crop = image[:, deep   : deep   + crop_size[0],
                                    height : height + crop_size[1],
                                    width  : width  + crop_size[2], :]
                image_input[:, :image_crop.shape[1], :image_crop.shape[2], :image_crop.shape[3], :] = image_crop

                pred, _ = model_func(image_input)
                if config.GAUSSIAN_SIGMA_COEFF > 0:
                    pred[0,:,:,:,0] = pred[0,:,:,:,0]*gaussian_kernel
                    pred[0,:,:,:,1] = pred[0,:,:,:,1]*gaussian_kernel
                    pred[0,:,:,:,2] = pred[0,:,:,:,2]*gaussian_kernel
                    pred[0,:,:,:,3] = pred[0,:,:,:,3]*gaussian_kernel
                #print(outputs[0].shape)
                #----------------Average-------------------------------
                whole_pred[:, deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2], :] += pred[:, :image_crop.shape[1], :image_crop.shape[2], :image_crop.shape[3], :]

                count_used[deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2]] += 1
    #print(whole_pred.shape)
    count_used = np.expand_dims(count_used, (0, -1))
    whole_pred = whole_pred / count_used
    #whole_pred = softmax(whole_pred, axis=-1)
    # final_pred = np.argmax(whole_pred, axis=-1)
    # #print(whole_pred.shape)

    # affine = np.array([
    #         [-1., -0., -0., 0.],
    #         [-0., -1., -0., 239.],
    #         [ 0., 0., 1., 0.],
    #         [ 0., 0., 0., 1.],
    #     ])

    # img = nib.Nifti1Image(final_pred, affine)
    # nib.save(img, os.path.join('../output', 'test.nii.gz'))  
    # print(f"writing to test.nii.gz: {time.time()-start:.02f} sec")

    return np.squeeze(whole_pred)

def segment_one_image(data, model_func, is_online=False):
    """
    perform inference and unpad the volume to original shape
    """
    img = data['images']
    temp_weight = data['weights'][:,:,:,0]
    temp_size = data['original_shape']
    temp_bbox = data['bbox']
    # Ensure online evaluation match the training patch shape...should change in future 
    batch_data_shape = config.PATCH_SIZE if is_online else config.INFERENCE_PATCH_SIZE
    
    img = img[np.newaxis, ...] # add batch dim

    im = img

    if config.MULTI_VIEW:
        im_ax = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_ax = transpose_volumes(im_ax, 'axial')
        prob1_ax = batch_segmentation(im_ax, model_func[0], data_shape=batch_data_shape)

        im_sa = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_sa = transpose_volumes(im_sa, 'sagittal')
        prob1_sa = batch_segmentation(im_sa, model_func[1], data_shape=batch_data_shape)

        im_co = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_co = transpose_volumes(im_co, 'coronal')
        prob1_co = batch_segmentation(im_co, model_func[2], data_shape=batch_data_shape)

        prob1 = (prob1_ax + np.transpose(prob1_sa, (1, 2, 0, 3)) + np.transpose(prob1_co, (1, 0, 2, 3))) / 3.0
        pred1 = np.argmax(prob1, axis=-1)
        
    else:
        im_pred = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_pred = transpose_volumes(im_pred, config.DIRECTION)
        # prob1 = batch_segmentation(im_pred, model_func[0], data_shape=batch_data_shape)
        prob1 = overlapping_inference(im_pred, model_func[0], data_shape=batch_data_shape)
        if config.DIRECTION == 'sagittal':
            prob1 = np.transpose(prob1, (1, 2, 0, 3))
        elif config.DIRECTION == 'coronal':
            prob1 = np.transpose(prob1, (1, 0, 2, 3))
        else:
            prob1 = prob1
        
        if config.NUM_CLASS == 1:
            pred1 = prob1 >= 0.5
            pred1 = np.squeeze(pred1, axis=-1)
        else:
            pred1 = np.argmax(prob1, axis=-1)
    
    pred1[pred1 == 3] = 4
    # pred1 should be the same as cropped brain region
    if config.ADVANCE_POSTPROCESSING:
        out_label = post_processing(pred1, temp_weight)
    else:
        out_label = pred1
    out_label = np.asarray(out_label, np.int16)

    if 'is_flipped' in data and data['is_flipped']:
        out_label = np.flip(out_label, axis=-1)
        prob1 = np.flip(prob1, axis=2) # d, h, w, num_class
    
    final_label = np.zeros(temp_size, np.int16)
    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)

    final_probs = np.zeros(list(temp_size) + [config.NUM_CLASS], np.float32)
    final_probs = set_ND_volume_roi_with_bounding_box_range(final_probs, temp_bbox[0]+[0], temp_bbox[1]+[config.NUM_CLASS - 1], prob1)
        
    return final_label, final_probs

def dice_of_brats_data_set(gt, pred, type_idx):
    dice_all_data = []
    for i in range(len(gt)):
        g_volume = copy.deepcopy(gt[i])
        s_volume = copy.deepcopy(pred[i])
        dice_one_volume = []
        if(type_idx ==0): # whole tumor
            if config.NUM_CLASS == 2:
                g_volume[g_volume == 4] = 1
                g_volume[g_volume == 2] = 1
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 1): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        else:
            #for label in [1, 2, 3, 4]: # dice of each class
            temp_dice = binary_dice3d(s_volume == 4, g_volume == 4)
            dice_one_volume = [temp_dice]
        if config.EVALUATE:
            print("idx="+str(i), dice_one_volume)
        dice_all_data.append(dice_one_volume)
    return dice_all_data


def post_processing_iseg(predict_label, weight_map):
    predict_label = predict_label * weight_map
    return predict_label

def eval_brats(df, detect_func, with_gt=True, save_nii=True, no_f1_hdd=False):
    """
    evaluation
    """
    df.reset_state()
    gts = []
    results = []
    gts_filename = []
    i = 0
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for filename, image_id, data in df.get_data():
            final_label, probs = detect_func(data)
            if config.TEST_FLIP:
                pred_flip, probs_flip = detect_func(flip_lr(data))
                final_prob = (probs + probs_flip) / 2.0
                pred = np.argmax(final_prob, axis=-1)
                pred[pred == 3] = 4
                if config.ADVANCE_POSTPROCESSING:
                    pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
                    pred = post_processing(pred, data['weights'][:,:,:,0])
                    pred = np.asarray(pred, np.int16)
                    final_label = np.zeros(data['original_shape'], np.int16)
                    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
                else:
                    final_label = pred
                if config.DATASET == 'iseg':
                    pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
                    pred = post_processing_iseg(pred, data['weights'][:,:,:,0]) # limit inside brain
                    pred = np.asarray(pred, np.int16)
                    final_label = np.zeros(data['original_shape'], np.int16)
                    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
                    final_label = change_brats_iseg_label(final_label, [2, 4, 1], [1, 2, 3]) ## iseg k cos loofng nhw trwsng gaf choox white matter vaf grey matter
            if save_nii:
                print(np.unique(final_label))
                save_to_nii(final_label, image_id, outdir=config.save_pred, mode="label")
            
            gts_filename.append("{}-label.img".format(filename) if config.DATASET == 'iseg' else "{}/{}_seg.nii.gz".format(filename, image_id))
            gt = load_nifty_volume_as_array("{}-label.img".format(filename) if config.DATASET == 'iseg' else "{}/{}_seg.nii.gz".format(filename, image_id), with_gt=True)
            # save_to_nii(gt, image_id + "-label", outdir=config.save_pred, mode="label")
            gts.append(gt)
            results.append(final_label)
            pbar.update()
            i = i + 1
            if config.DEBUG:
                if i == 1:
                    break
    dices = {}
    if 'brats' in config.DATASET:
        test_types = ['WT', 'TC', 'ET']
        class_num = 1 if config.NUM_CLASS == 1 else config.NUM_CLASS - 1
        for type_idx in range(class_num):
            dice = dice_of_brats_data_set(gts, results, type_idx)
            dice = np.asarray(dice)
            dice_mean = dice.mean(axis = 0)
            test_type = test_types[type_idx]
            dices[test_type] = dice_mean[0]
    elif 'iseg' in config.DATASET:
        test_types = ['CSF', 'GM', 'WM']
        class_num = config.NUM_CLASS if config.NUM_CLASS == 1 else config.NUM_CLASS - 1
        for type_idx in range(class_num):
            dice = dice_of_iseg_data_set(gts, results, type_idx)
            dice = np.asarray(dice)
            dice_mean = dice.mean(axis = 0)
            dice_std  = dice.std(axis = 0)
            test_type = test_types[type_idx]
            dices[test_type] = dice_mean[0]
    dicetemp, hdd, mhdd, asd = calculate_dice_hdd_asd(gts, results, gts_filename)
    return dices, hdd, mhdd, asd

def dice_of_iseg_data_set(gt, result, type_idx):
    dice_all_data = []
    for i in range(len(gt)):
        g_volume = copy.deepcopy(gt[i])
        s_volume = copy.deepcopy(result[i])
        dice_one_volume = []
        if(type_idx ==0): # CSF
            temp_dice = binary_dice3d(s_volume == 1, g_volume == 1)
            dice_one_volume = [temp_dice]
        elif(type_idx == 1): # GM
            temp_dice = binary_dice3d(s_volume == 2, g_volume == 2)
            dice_one_volume = [temp_dice]
        else: # WM
            #for label in [1, 2, 3, 4]: # dice of each class
            temp_dice = binary_dice3d(s_volume == 3, g_volume == 3)
            dice_one_volume = [temp_dice]
        dice_all_data.append(dice_one_volume)
    return dice_all_data

def pred_brats(df, detect_func):
    df.reset_state()
    results = []

    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for filename, image_id, data in df.get_data():
            final_label, probs = detect_func(data)
            if config.TEST_FLIP:
                pred_flip, probs_flip = detect_func(flip_lr(data))
                final_prob = (probs + probs_flip) / 2.0

                pred = np.argmax(final_prob, axis=-1)
                pred[pred == 3] = 4
                if config.ADVANCE_POSTPROCESSING:
                    pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
                    pred = post_processing(pred, data['weights'][:,:,:,0])
                    pred = np.asarray(pred, np.int16)
                    final_label = np.zeros(data['original_shape'], np.int16)
                    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
                else:
                    final_label = pred
            save_to_nii(final_label, image_id, outdir=config.save_pred, mode="label")
            pbar.update()
    return None


if __name__ == "__main__":
    from  data_loader import BRATS_SEG
    def get_eval_dataflow():
        if config.CROSS_VALIDATION:
            imgs = BRATS_SEG.load_from_file(config.BASEDIR, config.VAL_DATASET)
        else:
            if isinstance(config.BASEDIR, (list, tuple)):
                imgs = []
                for basedir in config.BASEDIR:
                    imgs.extend(BRATS_SEG.load_many(
                        basedir, config.VAL_DATASET, add_gt=False))

            else:
                imgs = BRATS_SEG.load_many(
                    config.BASEDIR, config.VAL_DATASET, add_gt=False)
        imgs = imgs[:1]
        # no filter for training
        if config.NO_CACHE:
            ds = DataFromListOfDict(imgs, ['file_name', 'id', 'image_data', 'gt'])
        else:
            ds = DataFromListOfDict(imgs, ['file_name', 'id', 'preprocessed'])

        def f(data):
            volume_list, label, weight, original_shape, bbox = data
            batch = sampler3d_whole(volume_list, label, weight, original_shape, bbox)
            return batch

        def preprocess(data):
            gt, im = data[-1], data[-2]
            volume_list, label, weight, original_shape, bbox = crop_brain_region(im, gt)
            batch = sampler3d_whole(volume_list, label, weight, original_shape, bbox)
            
            return [data[0], data[1], batch]
        
        if config.NO_CACHE:
            ds = MultiProcessMapDataZMQ(ds, nr_proc=1, map_func=preprocess, buffer_size=config.BATCH_SIZE, strict=True)
            #ds = MultiThreadMapData(ds, nr_thread=1, map_func=preprocess, buffer_size=4)
        else:
            ds = MapDataComponent(ds, f, 2)
        #ds = PrefetchDataZMQ(ds, 1)
        return ds
    from tensorpack import *
    pred_func = OfflinePredictor(PredictConfig(
                model=get_model(modelType="inference"),
                session_init=get_model_loader(args.load) if args.load else None,
                input_names=['image'],
                output_names=get_model_output_names()))
    # autotune is too slow for inference
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    df = get_eval_dataflow()
    # if config.DYNAMIC_SHAPE_PRED:    
    # eval_brats(df, lambda img: segment_one_image_dynamic(img, pred_func))
    # else:
    result = eval_brats(df, lambda img: segment_one_image(img, pred_func))
    print(str(result))