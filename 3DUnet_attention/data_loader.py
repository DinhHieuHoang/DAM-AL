#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: data_loader.py

import numpy as np
from termcolor import colored
from tabulate import tabulate

from tensorpack.utils import logger
from tensorpack.utils.rect import FloatBox
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

import random, pickle, glob, os
from tqdm import tqdm
from utils import crop_brain_region
import config

class BRATS_SEG(object):
    def __init__(self, basedir, mode):
        """
        basedir="/data/dataset/BRATS2018/{mode}/{HGG/LGG}/patient_id/{flair/t1/t1ce/t2/seg}"
        mode: training/val/test
        """
        print(basedir, mode)
        self.basedir = os.path.join(basedir, mode)
        self.mode = mode

    def load_kfold(self):
        with open(config.CROSS_VALIDATION_PATH, 'rb') as f:
            data = pickle.load(f)
        imgs = data["fold{}".format(config.FOLD)][self.mode]
        patient_ids = [x.split("/")[-1] for x in imgs]
        ret = []
        print("Preprocessing {} Data ...".format(self.mode))
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
            data = {}
            data['image_data'] = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            # read modality
            mod = glob.glob(file_name+"/*.nii*")
            assert len(mod) >= 4  # 4mod +1gt
            for m in mod:
                if 'seg' in m:
                    data['gt'] = m
                else:
                    _m = m.split("/")[-1].split(".")[0].split("_")[-1]
                    data['image_data'][_m] = m
            if 'gt' in data:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
            ret.append(data)
        return ret

    def load_pancreas(self):
        return None

    def load_iseg(self):
        # subject-10-label.hdr  subject-2-label.hdr  subject-4-label.hdr  subject-6-label.hdr  subject-8-label.hdr
        # subject-10-label.img  subject-2-label.img  subject-4-label.img  subject-6-label.img  subject-8-label.img
        # subject-10-T1.hdr     subject-2-T1.hdr     subject-4-T1.hdr     subject-6-T1.hdr     subject-8-T1.hdr
        # subject-10-T1.img     subject-2-T1.img     subject-4-T1.img     subject-6-T1.img     subject-8-T1.img
        # subject-10-T2.hdr     subject-2-T2.hdr     subject-4-T2.hdr     subject-6-T2.hdr     subject-8-T2.hdr
        # subject-10-T2.img     subject-2-T2.img     subject-4-T2.img     subject-6-T2.img     subject-8-T2.img
        # subject-1-label.hdr   subject-3-label.hdr  subject-5-label.hdr  subject-7-label.hdr  subject-9-label.hdr
        # subject-1-label.img   subject-3-label.img  subject-5-label.img  subject-7-label.img  subject-9-label.img
        # subject-1-T1.hdr      subject-3-T1.hdr     subject-5-T1.hdr     subject-7-T1.hdr     subject-9-T1.hdr
        # subject-1-T1.img      subject-3-T1.img     subject-5-T1.img     subject-7-T1.img     subject-9-T1.img
        # subject-1-T2.hdr      subject-3-T2.hdr     subject-5-T2.hdr     subject-7-T2.hdr     subject-9-T2.hdr
        # subject-1-T2.img      subject-3-T2.img     subject-5-T2.img     subject-7-T2.img     subject-9-T2.img
        self.basedir = '/media/Seagate16T/datasets/iSeg2019/iSeg-2019-Training/'
        self.traindir= '/home/dghan/iseg_2017/iSeg-2017-Training/'
        # self.testdir = '/home/dghan/iseg_2017/iSeg-2017-Validation/'
        self.testdir = '/home/dghan/iseg_2017/iSeg-2017-Testing/'
        # self.testdir = '/media/Seagate16T/datasets/iSeg2019/iSeg-2019-Training/'
        modalities = ['T1', 'T2']
        if 'training' in self.mode.lower():
            self.basedir = self.traindir
            # subject_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            subject_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '10']
            print("Data Folder (training): ", self.basedir)
        else:
            self.basedir = self.testdir
            if 'val' in self.mode.lower():
                subject_ids = [str(i) for i in range(11, 24)]
                # subject_ids = [str(i) for i in range(11, 15)] # gpu0
                # subject_ids = [str(i) for i in range(15, 18)] # gpu1
                # subject_ids = [str(i) for i in range(18, 21)] # gpu2
                # subject_ids = [str(i) for i in range(21, 24)] # gpu3
                # subject_ids = ['11']
                # subject_ids = ['9']
                print("Data Folder (validation): ", self.basedir)
            else:
                if 'test' in self.mode.lower():
                    assert 1 == 2 , 'test path for iseg did not defined'
        ret = []
        for subject_id in subject_ids:
            data = {}
            data['file_name'] = self.basedir+'subject-'+subject_id
            data['id'] = subject_id
            data['gt'] = data['file_name'] + '-label.img'
            data['image_data'] = {
                't1': data['file_name'] + '-T1.img',
                't2': data['file_name'] + '-T2.img'
            }

            if 'gt' in data:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
            else:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                    del data['image_data']
            ret.append(data)
        return ret

    def load_3d_2020(self):
        """
        dataset_mode: none
        modalities: ['flair', 't1ce', 't1', 't2']
        """
        print("Data Folder: ", self.basedir)

        modalities = ['flair', 't1ce', 't1', 't2']
        
        imgs = glob.glob(self.basedir+"/*")
        imgs = [x for x in imgs if '.csv' not in x]
        patient_ids = [x.split("/")[-1] for x in imgs]

        ret = []
        print("Preprocessing Data ...")
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
        # for idx in range(2 if config.DEBUG else len(imgs)):
            file_name = imgs[idx]
            data = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            data['gt'] = file_name + '/' + data['id'] + '_seg.nii.gz'
            data['image_data'] = {
                'flair': file_name + '/' + data['id'] + '_flair.nii.gz',
                't1ce': file_name + '/' + data['id'] + '_t1ce.nii.gz',
                't1': file_name + '/' + data['id'] + '_t1.nii.gz',
                't2': file_name + '/' + data['id'] + '_t2.nii.gz'
            }
            print(data)
            if 'gt' in data:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
            else:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                    del data['image_data']
            ret.append(data)
        return ret

    def load_3d(self):
        """
        dataset_mode: HGG/LGG/ALL
        return list(dict[patient_id][modality] = filename.nii.gz)
        """
        print("Data Folder: ", self.basedir)

        modalities = ['flair', 't1ce', 't1.', 't2']
        
        if 'training' in self.basedir.lower():
            img_HGG = glob.glob(self.basedir+"/HGG/*")
            img_LGG = glob.glob(self.basedir+"/LGG/*")
            imgs = img_HGG + img_LGG
        else:
            imgs = glob.glob(self.basedir+"/*")
        imgs = [x for x in imgs if '.csv' not in x]
        #imgs = imgs[:30]
        # imgs = ["/BraTS/BraTS_data/MICCAI_BraTS2020_ValidationData/val/BraTS20_Validation_069"]

        patient_ids = [x.split("/")[-1] for x in imgs]

        # print(patient_ids)

        ret = []
        print("Preprocessing Data ...")
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
            data = {}
            data['image_data'] = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            # read modality
            mod = glob.glob(file_name+"/*.nii*")
            # print("file_name  : ", file_name)
            # print("\npatient_ids[idx] ",patient_ids[idx] ,"\nfile_name ",file_name ,"\nmod ",mod)
            
            assert len(mod) >= 4, '{}'.format(file_name)  # 4mod +1gt        
            for m in mod:
                if 'seg' in m:
                    data['gt'] = m
                else:
                    _m = m.split("/")[-1].split(".")[0].split("_")[-1]
                    data['image_data'][_m] = m
            # print("data['image_data'] ",data['image_data'])
            # print("self.basedir ", self.basedir)

            if 'gt' in data:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
                # if config.GT_TEST:
                #     data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                #     del data['image_data']
            else:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                    del data['image_data']
            ret.append(data)
        
        return ret

    @staticmethod
    def load_from_file(basedir, names):
        brats = BRATS_SEG(basedir, names)
        return  brats.load_kfold()

    @staticmethod
    def load_many(basedir,names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            brats = BRATS_SEG(basedir, n)
            if config.DATASET == 'brats2020':
                ret.extend(brats.load_3d_2020())
            else:
                if config.DATASET == 'brats2019':
                    ret.extend(brats.load_3d())
                else:
                    if config.DATASET == 'iseg':
                        ret.extend(brats.load_iseg())
                    else:
                        if config.DATASET == 'pancreas':
                            ret.extend(brats.load_pancreas())
        return ret
