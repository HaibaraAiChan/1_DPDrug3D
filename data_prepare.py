import sys
import os
import argparse

import numpy as np

from deepdrug3d import DeepDrug3DBuilder

from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils
import gc
import cPickle
import shutil


def pre_pro_data(pos_list, neg_list, voxel_folder, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    voxel_name_list = []
    for filename in os.listdir(voxel_folder):
        if filename:
            voxel_name_list.append(filename[0:-4])

    pos = []
    with open(pos_list) as atp_in:
        for line in atp_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            pos.append(temp)
    neg = []
    with open(neg_list) as heme_in:
        for line in heme_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            neg.append(temp)
    # convert data into a single matrix
    pos_len = len(pos)
    neg_len = len(neg)
    L = pos_len + neg_len

    print('...list pos')
    for p1 in pos:
        pp = p1.split('_')
        pocket1 = pp[0] + '_' + pp[1]
        pocket2 = pp[2] + '_' + pp[3]
        assert any(pocket1 in voxel for voxel in voxel_name_list)
        assert any(pocket2 in voxel for voxel in voxel_name_list)

        full_path_1 = voxel_folder + '/' + pocket1 + '.pkl'
        full_path_2 = voxel_folder + '/' + pocket2 + '.pkl'

        temp1 = np.load(full_path_1)
        temp2 = np.load(full_path_2)
        temp = np.append(temp1, temp2, axis=1)

        print pocket1 + ' ' + pocket2

        oname = output_folder + p1 + '_' + str(int(1)) + ".pkl"
        cPickle.dump(temp, open(oname, "wb"))

    print('...List neg')
    for p1 in neg:
        pp = p1.split('_')
        pocket1 = pp[0] + '_' + pp[1]
        pocket2 = pp[2] + '_' + pp[3]
        assert any(pocket1 in voxel for voxel in voxel_name_list)
        assert any(pocket2 in voxel for voxel in voxel_name_list)

        full_path_1 = voxel_folder + pocket1 + '.pkl'
        full_path_2 = voxel_folder + pocket2 + '.pkl'

        temp1 = np.load(full_path_1)
        temp2 = np.load(full_path_2)
        temp = np.append(temp1, temp2, axis=1)

        print pocket1 + ' ' + pocket2
        oname = output_folder + p1 + '_' + str(int(0)) + ".pkl"
        cPickle.dump(temp, open(oname, "wb"))


def pre_pro_valid_data(pos_list, neg_list, voxel_folder, output_folder, valid_num):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    voxel_name_list = []
    for filename in os.listdir(voxel_folder):
        if filename:
            voxel_name_list.append(filename[0:-4])

    pos = []
    pos_num = 0
    with open(pos_list) as atp_in:
        for line in atp_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            pos.append(temp)
            pos_num = pos_num + 1
            if pos_num >= valid_num:
                break

    neg = []
    neg_num=0
    with open(neg_list) as heme_in:
        for line in heme_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            neg.append(temp)
            neg_num = neg_num + 1
            if neg_num >= valid_num:
                break
    # convert data into a single matrix
    pos_len = len(pos)
    neg_len = len(neg)
    L = pos_len + neg_len

    print('...list pos')
    for p1 in pos:
        pp = p1.split('_')
        pocket1 = pp[0] + '_' + pp[1]
        pocket2 = pp[2] + '_' + pp[3]
        assert any(pocket1 in voxel for voxel in voxel_name_list)
        assert any(pocket2 in voxel for voxel in voxel_name_list)

        full_path_1 = voxel_folder + '/' + pocket1 + '.pkl'
        full_path_2 = voxel_folder + '/' + pocket2 + '.pkl'

        temp1 = np.load(full_path_1)
        temp2 = np.load(full_path_2)
        temp = np.append(temp1, temp2, axis=1)

        print pocket1 + ' ' + pocket2

        oname = output_folder + p1 + '_d_' + str(int(1)) + ".pkl"
        cPickle.dump(temp, open(oname, "wb"))

    print('...List neg')
    for p1 in neg:
        pp = p1.split('_')
        pocket1 = pp[0] + '_' + pp[1]
        pocket2 = pp[2] + '_' + pp[3]
        assert any(pocket1 in voxel for voxel in voxel_name_list)
        assert any(pocket2 in voxel for voxel in voxel_name_list)

        full_path_1 = voxel_folder + pocket1 + '.pkl'
        full_path_2 = voxel_folder + pocket2 + '.pkl'

        temp1 = np.load(full_path_1)
        temp2 = np.load(full_path_2)
        temp = np.append(temp1, temp2, axis=1)

        print pocket1 + ' ' + pocket2
        oname = output_folder + p1 + '_d_' + str(int(0)) + ".pkl"
        cPickle.dump(temp, open(oname, "wb"))


if __name__ == "__main__":
    pos_list = './pos-new-ATP.lst'
    neg_list = './neg-new-ATP.lst'
    voxel_folder = './voxel_output/'
    train_folder = './train_data/'
    valid_folder = './valid_data/'

    # pre_pro_data(pos_list, neg_list, voxel_folder, train_folder)

    valid_num = 10

    pre_pro_valid_data(pos_list, neg_list, voxel_folder, valid_folder, valid_num/2)
