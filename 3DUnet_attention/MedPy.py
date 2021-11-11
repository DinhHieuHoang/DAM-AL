# # old before some bug in final eval
# #!/usr/bin/env python
# # coding: utf-8
# import time
# import numpy as np
# import os
# import nibabel as nib
# from medpy import metric
# from scipy.spatial.distance import directed_hausdorff

# def newhdd(gt, pred):
#     r=[]
#     for j in range(0,gt.shape[2]):
#         m=max(directed_hausdorff(gt[:,:,j],pred[:,:,j])[0],directed_hausdorff(pred[:,:,j],gt[:,:,j])[0])
#         r.append(m)
#     return np.max(r)

# def preprocess_label(label):
#     bachground = label == 0
#     ncr = label == 1 # Necrotic and Non-Enhancing Tumor (NCR/NET)
#     ed = label == 2 # Peritumoral Edema (ED)
#     et = label == 4 # GD-enhancing Tumor (ET)
#     WT = ncr+ed+et
#     TC = ncr + et
#     ET = et
#     return np.array([bachground, WT, TC, ET], dtype=np.uint8)

# def f1_hdd_3d(gt,pred):
#     t1 = time.time()
#     f1 = {}
#     hdd = {}
    
#     for c in np.unique(gt):
#         mask_gt = gt == c
#         mask_pred = pred == c
#         if c in np.unique(pred):
#             dc = metric.binary.dc(mask_pred,mask_gt)
#             hd = metric.binary.hd95(mask_pred,mask_gt)
# #             hd = newhdd(mask_pred,mask_gt)
#         else:
#             dc = 0.0
#             hd = mask_gt.shape[0]
# #             hd = newhdd(mask_pred,mask_gt)
#         f1[int(c)] = dc
#         hdd[int(c)] = hd
#     print('Processing Time:',time.time()-t1)
#     return f1,hdd

# def compute_f1_hdd_3d(mask_gt,mask_pred):
#     t1 = time.time()
#     idx2class = {0:'Background',1:'WT',2:'TC',3:'ET'}
#     f1 = {}
#     hdd = {}
    
#     for i in range(1,mask_gt.shape[0]):
#         try:
#             dc = metric.binary.dc(mask_pred[i,:,:,:],mask_gt[i,:,:,:])
#             f1[idx2class[int(i)]] = dc
#         except:
#             f1[idx2class[int(i)]] = None
#         try:
#             if np.unique(mask_pred[i,:,:,:]).shape[0] == 1 and False in np.unique(mask_pred[i,:,:,:]>0):
#                 mask_pred[i,:,:,:] = np.ones_like(mask_pred[i,:,:,:])
# #               mask_gt[i,:,:,:] = ~(mask_gt[i,:,:,:] == 1)
#             hd = metric.binary.hd95(mask_gt[i,:,:,:],mask_pred[i,:,:,:])
# #             hd = newhdd(mask_pred,mask_gt) 
#             hdd[idx2class[int(i)]] = hd           
#         except:
#             hdd[idx2class[int(i)]] = None
        
#     print('Processing Time:',time.time()-t1)
#     return f1, hdd

# #@dghan use this function:
# def calculate_f1_hdd(gt, pred, gts_filename = None):
#     f1  = {'WT': [], 'TC': [], 'ET': []}
#     hdd = {'WT': [], 'TC': [], 'ET': []}
#     l = len(gt)
#     for i in range(l):
#         pgt = preprocess_label(gt[i])
#         ppred = preprocess_label(pred[i])
#         f1_i, hdd_i = compute_f1_hdd_3d(pgt,ppred)
#         if gts_filename is not None:
#             print(str(gts_filename[i]) + '\t' + str(f1_i) + '\t' + str(hdd_i))
#         if f1_i['WT'] is not None: f1['WT'].append(np.mean(f1_i['WT']))
#         if f1_i['TC'] is not None: f1['TC'].append(np.mean(f1_i['TC']))
#         if f1_i['ET'] is not None: f1['ET'].append(np.mean(f1_i['ET']))
#         if hdd_i['WT'] is not None: hdd['WT'].append(np.mean(hdd_i['WT']))
#         if hdd_i['TC'] is not None: hdd['TC'].append(np.mean(hdd_i['TC']))
#         if hdd_i['ET'] is not None: hdd['ET'].append(np.mean(hdd_i['ET']))
#     f1['WT'] = np.mean(f1['WT'])
#     f1['TC'] = np.mean(f1['TC'])
#     f1['ET'] = np.mean(f1['ET'])
#     hdd['WT'] = np.mean(hdd['WT'])
#     hdd['TC'] = np.mean(hdd['TC'])
#     hdd['ET'] = np.mean(hdd['ET'])
#     return f1, hdd

# def f1_numpy(gt,pred):
#     t1 = time.time()
#     f1 = {}
    
#     for c in np.unique(gt):
#         if c in np.unique(pred):
#             mask_gt = gt == c
#             mask_pred = pred == c
#             f1[int(c)] = 2*(np.sum(mask_gt & mask_pred))/(np.sum(mask_gt)+np.sum(mask_pred))
#     print('Processing Time:',time.time()-t1)
#     return f1


#!/usr/bin/env python
# coding: utf-8
import time
import config
import numpy as np
import os
import nibabel as nib
from medpy import metric
from scipy.spatial.distance import directed_hausdorff

def newhdd(gt, pred):
    r=[]
    for j in range(0,gt.shape[2]):
        m=max(directed_hausdorff(gt[:,:,j],pred[:,:,j])[0],directed_hausdorff(pred[:,:,j],gt[:,:,j])[0])
        r.append(m)
    return np.max(r)

def preprocess_label(label):
    if config.DATASET == 'iseg':
        return np.array([label==0, label==1, label==2, label==3])
    bachground = label == 0
    ncr = label == 1 # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = label == 2 # Peritumoral Edema (ED)
    et = label == 4 # GD-enhancing Tumor (ET)
    WT = ncr+ed+et
    TC = ncr + et
    ET = et
    return np.array([bachground, WT, TC, ET], dtype=np.uint8)

def f1_hdd_3d(gt,pred):
    t1 = time.time()
    f1 = {}
    hdd = {}
    
    for c in np.unique(gt):
        mask_gt = gt == c
        mask_pred = pred == c
        if c in np.unique(pred):
            dc = metric.binary.dc(mask_pred,mask_gt)
            hd = metric.binary.hd95(mask_pred,mask_gt)
        else:
            dc = 0.0
            hd = mask_gt.shape[0]
        f1[int(c)] = dc
        hdd[int(c)] = hd
    print('Processing Time:',time.time()-t1)
    return f1,hdd

def compute_f1_hdd_3d(mask_gt,mask_pred):
    label = list(config.idx2class.values())
    f1 = {}
    hdd = {}
    
    for i in range(1,mask_gt.shape[0]):
        try: # dice score
            dc = metric.binary.dc(mask_pred[i,:,:,:],mask_gt[i,:,:,:])
            f1[label[i]] = dc
        except:
            f1[label[i]] = None
        try: # hdd
            if np.unique(mask_pred[i,:,:,:]).shape[0] == 1 and False in np.unique(mask_pred[i,:,:,:]>0): # only 1 label which is 0 (background)
                mask_pred[i,:,:,:] = np.ones_like(mask_pred[i,:,:,:])
            hd = metric.binary.hd(mask_gt[i,:,:,:],mask_pred[i,:,:,:])
            # hd = metric.binary.asd(mask_gt[i,:,:,:],mask_pred[i,:,:,:])
            # hd = metric.binary.hd95(np.ones_like(mask_pred[i,:,:,:]), mask_gt[i,:,:,:])
            print((mask_gt[i,:,:,:] == mask_pred[i,:,:,:]).all())
            print(hd)
            hdd[label[i]] = hd           
        except:
            hdd[label[i]] = None
    return f1, hdd

#@dghan use this function:
def calculate_f1_hdd(gt, pred, gts_filename = None):
    label = list(config.idx2class.values())
    f1  = {label[1]: [], label[2]: [], label[3]: []}
    hdd = {label[1]: [], label[2]: [], label[3]: []}
    l = len(gt)
    for i in range(l):
        pgt = preprocess_label(gt[i])
        ppred = preprocess_label(pred[i])
        f1_i, hdd_i = compute_f1_hdd_3d(pgt,ppred)
        if gts_filename is not None:
            print(str(gts_filename[i]) + '\t' + str(f1_i) + '\t' + str(hdd_i))
            print((np.mean(f1[label[2]])-np.mean(f1[label[1]])) == 0)
            print((np.mean(hdd[label[2]]) - np.mean(hdd[label[1]])) == 0)
        if f1_i[label[1]] is not None: f1[label[1]].append(np.mean(f1_i[label[1]]))
        if f1_i[label[2]] is not None: f1[label[2]].append(np.mean(f1_i[label[2]]))
        if f1_i[label[3]] is not None: f1[label[3]].append(np.mean(f1_i[label[3]]))
        if hdd_i[label[1]] is not None: hdd[label[1]].append(np.mean(hdd_i[label[1]]))
        if hdd_i[label[2]] is not None: hdd[label[2]].append(np.mean(hdd_i[label[2]]))
        if hdd_i[label[3]] is not None: hdd[label[3]].append(np.mean(hdd_i[label[3]]))
    f1[label[1]] = np.mean(f1[label[1]])
    f1[label[2]] = np.mean(f1[label[2]])
    f1[label[3]] = np.mean(f1[label[3]])
    hdd[label[1]] = np.mean(hdd[label[1]])
    hdd[label[2]] = np.mean(hdd[label[2]])
    hdd[label[3]] = np.mean(hdd[label[3]])
    return f1, hdd

def compute_dice_hdd_asd_3d(mask_gt, mask_pred):
    label = list(config.idx2class.values())
    dice = {}
    hdd = {}
    mhdd= {}
    asd = {}
    
    for i in range(1,mask_gt.shape[0]):
        try: # dice score
            dc = metric.binary.dc(mask_pred[i,:,:,:],mask_gt[i,:,:,:])
            dice[label[i]] = dc
        except:
            dice[label[i]] = None
        try: # hdd + mhdd
            if np.unique(mask_pred[i,:,:,:]).shape[0] == 1 and False in np.unique(mask_pred[i,:,:,:]>0): # only 1 label which is 0 (background)
                mask_pred[i,:,:,:] = np.ones_like(mask_pred[i,:,:,:])
            # hd = metric.binary.hd95(mask_pred[i,:,:,:], mask_gt[i,:,:,:])
            hd = metric.binary.hd(mask_gt[i,:,:,:],mask_pred[i,:,:,:])
            hd1 = metric.binary.__surface_distances(mask_gt[i,:,:,:], mask_pred[i,:,:,:], voxelspacing=None, connectivity=1)
            hd2 = metric.binary.__surface_distances(mask_pred[i,:,:,:], mask_gt[i,:,:,:], voxelspacing=None, connectivity=1)
            hdstack = np.hstack((hd1, hd2))
            hdstack = hdstack[hdstack > 1]
            hdd[label[i]] = hd
            mhdd[label[i]] = np.percentile(hdstack, 95)
        except:
            hdd[label[i]] = None
            mhdd[label[i]]= None
        try: # asd
            sd = metric.binary.asd(mask_gt[i,:,:,:],mask_pred[i,:,:,:])
            asd[label[i]] = sd           
        except:
            asd[label[i]] = None
    return dice, hdd, mhdd, asd

def calculate_dice_hdd_asd(gt, pred, gts_filename = None):
    label = list(config.idx2class.values())
    dice  = {label[1]: [], label[2]: [], label[3]: []}
    hdd = {label[1]: [], label[2]: [], label[3]: []}
    mhdd = {label[1]: [], label[2]: [], label[3]: []}
    asd = {label[1]: [], label[2]: [], label[3]: []}
    l = len(gt)
    for i in range(l):
        pgt = preprocess_label(gt[i])
        ppred = preprocess_label(pred[i])
        dice_i, hdd_i, mhdd_i, asd_i = compute_dice_hdd_asd_3d(pgt,ppred)
        if gts_filename is not None:
            print(str(gts_filename[i]) + '\t' + str(dice_i) + '\t' + str(hdd_i) + '\t' + str(mhdd_i) + '\t' + str(asd_i))
            # print((np.mean(dice[label[2]])-np.mean(dice[label[1]])) == 0)
            # print((np.mean(hdd[label[2]]) - np.mean(hdd[label[1]])) == 0)
        if dice_i[label[1]] is not None: dice[label[1]].append(np.mean(dice_i[label[1]]))
        if dice_i[label[2]] is not None: dice[label[2]].append(np.mean(dice_i[label[2]]))
        if dice_i[label[3]] is not None: dice[label[3]].append(np.mean(dice_i[label[3]]))
        if hdd_i[label[1]] is not None: hdd[label[1]].append(np.mean(hdd_i[label[1]]))
        if hdd_i[label[2]] is not None: hdd[label[2]].append(np.mean(hdd_i[label[2]]))
        if hdd_i[label[3]] is not None: hdd[label[3]].append(np.mean(hdd_i[label[3]]))
        if mhdd_i[label[1]] is not None: mhdd[label[1]].append(np.mean(mhdd_i[label[1]]))
        if mhdd_i[label[2]] is not None: mhdd[label[2]].append(np.mean(mhdd_i[label[2]]))
        if mhdd_i[label[3]] is not None: mhdd[label[3]].append(np.mean(mhdd_i[label[3]]))
        if asd_i[label[1]] is not None: asd[label[1]].append(np.mean(asd_i[label[1]]))
        if asd_i[label[2]] is not None: asd[label[2]].append(np.mean(asd_i[label[2]]))
        if asd_i[label[3]] is not None: asd[label[3]].append(np.mean(asd_i[label[3]]))
    dice[label[1]] = np.mean(dice[label[1]])
    dice[label[2]] = np.mean(dice[label[2]])
    dice[label[3]] = np.mean(dice[label[3]])
    hdd[label[1]] = np.mean(hdd[label[1]])
    hdd[label[2]] = np.mean(hdd[label[2]])
    hdd[label[3]] = np.mean(hdd[label[3]])
    mhdd[label[1]] = np.mean(mhdd[label[1]])
    mhdd[label[2]] = np.mean(mhdd[label[2]])
    mhdd[label[3]] = np.mean(mhdd[label[3]])
    asd[label[1]] = np.mean(asd[label[1]])
    asd[label[2]] = np.mean(asd[label[2]])
    asd[label[3]] = np.mean(asd[label[3]])
    return dice, hdd, mhdd, asd

# if __name__ == '__main__':
