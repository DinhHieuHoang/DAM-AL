import utils
import nibabel
import numpy as np
import random
import os
import config
# import SimpleITK as sitk
import pickle
from scipy import ndimage
import copy

# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

traindir= '../../iseg_2017/iSeg-2017-Validation/'
testdir = '../../iseg_2017/iSeg-2017-Validation/'
preddir = './save_eval/train-log-debug/'

# (256, 192, 144)

def save_to_nii(im, filename, outdir="", mode="image", system="nibabel"):
    """
    Save numpy array to nii.gz format to submit
    im: 3d numpy array ex: [155, 240, 240]
    """
    if system == "sitk":
        if mode == 'label':
            img = sitk.GetImageFromArray(im.astype(np.uint8))
        else:
            img = sitk.GetImageFromArray(im.astype(np.float32))
        if not os.path.exists("./{}".format(outdir)):
            os.makedirs("./{}".format(outdir), exist_ok = True)
        sitk.WriteImage(img, "./{}/{}.nii.gz".format(outdir, filename))
    elif system == "nibabel":
        # img = np.rot90(im, k=2, axes= (1,2))
        img = np.moveaxis(im, 0, -1)
        img = np.moveaxis(img, 0, 1)
        OUTPUT_AFFINE = np.array(
                [[ -1,-0,-0,-0],
                 [ -0,-1,-0,239],
                 [  0, 0, 1,0],
                 [  0, 0, 0,1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists("./{}".format(outdir)):
            os.makedirs("./{}".format(outdir), exist_ok = True)
        nibabel.save(img, "./{}/{}.nii.gz".format(outdir, filename))
    else:
        img = np.rot90(im, k=2, axes= (1,2))
        OUTPUT_AFFINE = np.array(
                [[0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists("./{}".format(outdir)):
            os.makedirs("./{}".format(outdir), exist_ok = True)
        nibabel.save(img, "./{}/{}.nii.gz".format(outdir, filename))

def _cal_signed_distance_map(posmask):
    # given positive mask, calculate corresponding signed distance map 
    # return has the same shape with that of the input
    negmask = ~posmask
    posdis = ndimage.distance_transform_edt(posmask)
    negdis = ndimage.distance_transform_edt(negmask)
    res = negdis * np.array(negmask, dtype=np.float)
    res = res - (posdis - 1.0) * np.array(posmask, dtype=np.float)
    return res

def signed_distance_map(ground_truth):
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

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    # print(img.header)
    data = img.get_data()
    if len(data.shape) == 4:
        x, y, z, _ = data.shape
    else:
        x, y, z = data.shape
    data.resize(x,y,z)
    del x, y, z
    data = np.transpose(data, [2,1,0])
    del img
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

if __name__ == '__main__':
    # training subject
    # subject_id = '9'
    # data={}
    # data['file_name'] = traindir+'subject-'+subject_id
    # data['id'] = subject_id
    # data['gt'] = data['file_name'] + '-label.img'
    # data['image_data'] = {
    #     't1': data['file_name'] + '-T1.img',
    #     't2': data['file_name'] + '-T2.img'
    # }
    # training_img = load_nifty_volume_as_array(data['image_data']['t1'], with_header=False)
    # # training_img = load_nifty_volume_as_array(data['gt'], with_header=False)
    # print(training_img.shape)

    subject_id = '9'
    data={}
    data['file_name'] = testdir+'subject-'+subject_id
    data['id'] = subject_id
    data['gt'] = data['file_name'] + '-label.img'
    data['pred'] = preddir + subject_id + '.nii.gz'
    data['image_data'] = {
        't1': data['file_name'] + '-T1.img',
        't2': data['file_name'] + '-T2.img'
    }
    test_img = load_nifty_volume_as_array(data['image_data']['t1'], with_header=False)
    # test_img = load_nifty_volume_as_array(data['gt'], with_header=False)
    # test_img = utils.change_brats_iseg_label(test_img, [10, 150, 250], [1, 2, 3])
    print(test_img.shape)

    pred_img = load_nifty_volume_as_array(data['pred'], with_header=False)
    print(pred_img.shape)

    gr_img = load_nifty_volume_as_array(data['gt'], with_header=False)
    gr_img = utils.change_brats_iseg_label(gr_img, [10, 150, 250], [1, 2, 3])
    print(gr_img.shape)
    save_to_nii(im = gr_img, filename="label-relabel", outdir= testdir)

    


    # weight_map = modified_distance_map(gr_img)

    # weight_map_slide = weight_map[0][149]
    # plt.subplot()
    # sb.heatmap(weight_map_slide, square=True)
    # plt.show()

    # weight_map_slide = weight_map[1][149]
    # # plt.subplots(figsize =  (192, 144))
    # plt.subplot()
    # sb.heatmap(weight_map_slide, square=True)
    # plt.show()

    # weight_map_slide = weight_map[2][149]
    # plt.subplot()
    # sb.heatmap(weight_map_slide, square=True)
    # plt.show()