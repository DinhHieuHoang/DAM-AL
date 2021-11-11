import pickle
import numpy as np
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='training data path', default="../datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/")
    parser.add_argument('--out', help="output path", default="./5folds")
    parser.add_argument('--nfolds', type=int, help="number of folds", default=5)
    parser.add_argument('--year', type=int, help="different path for 2020 BraTS", default=2020)
    args = parser.parse_args()
    data = {}

    n_folds = args.nfolds

    if args.year == 2020:
        filenames = glob.glob(args.data+"*")
        filenames = [x for x in filenames if '.csv' not in x]
    else:
        filenames = glob.glob(args.data+"HGG/*")
    print(len(filenames))
    val_length = len(filenames) // n_folds
    np.random.shuffle(filenames)

    folds = []
    for i in range(n_folds):
        if i < n_folds - 1:
            folds.append(filenames[(val_length * i):(val_length * (i + 1))])
        else:
            folds.append(filenames[(val_length * i):])

    for i in range(n_folds):
        data['fold{}'.format(i)] = {}
        data['fold{}'.format(i)]['val'] = folds[i]

        data['fold{}'.format(i)]['training'] = []
        for j in range(n_folds):
            if j == i:
                continue
            else:
                data['fold{}'.format(i)]['training'] +=  folds[j]
        print(len(data['fold{}'.format(i)]['val']), len(data['fold{}'.format(i)]['training']))

    with open(args.out+".pkl", 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)




