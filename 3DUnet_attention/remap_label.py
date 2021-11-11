import utils
import nibabel
import numpy as np
import random
import os
import SimpleITK as sitk
import pickle
from scipy import ndimage
import copy

traindir= '/media/Seagate16T/datasets/iSeg2019/iSeg-2019-Training/'
testdir = '/home/dghan/iseg_2017/iSeg-2017-Validation/'

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
    subject_id = '9'
    data={}
    data['file_name'] = traindir+'subject-'+subject_id
    data['id'] = subject_id
    data['gt'] = data['file_name'] + '-label.img'
    data['image_data'] = {
        't1': data['file_name'] + '-T1.img',
        't2': data['file_name'] + '-T2.img'
    }
    training_img = load_nifty_volume_as_array(data['image_data']['t1'], with_header=False)
    # training_img = load_nifty_volume_as_array(data['gt'], with_header=False)
    print(training_img.shape)

    subject_id = '9'
    data={}
    data['file_name'] = testdir+'subject-'+subject_id
    data['id'] = subject_id
    data['gt'] = data['file_name'] + '-label.img'
    data['image_data'] = {
        't1': data['file_name'] + '-T1.img',
        't2': data['file_name'] + '-T2.img'
    }
    test_img = load_nifty_volume_as_array(data['image_data']['t1'], with_header=False)
    # test_img = load_nifty_volume_as_array(data['gt'], with_header=False)
    # test_img = utils.change_brats_iseg_label(test_img, [10, 150, 250], [1, 2, 3])
    print(test_img.shape)

    print((training_img == test_img).all())
    print(training_img[129][97])
    print(test_img[129][97])

