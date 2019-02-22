"""
the pos-new-ATP-old.lst
the neg-new-ATP.lst
contain many voxel data, make sure they can be found in
the voxel data-set


many XXX in neg-new-ATP.lst have to be deleted
"""

import os

pos = './list/pos-new-ATP-old.lst'
neg = './list/neg-new-ATP-old.lst'
voxel_folder = './voxel_output/'

list_pos = []
list_neg = []
voxel_name_list = []
for filename in os.listdir(voxel_folder):
    if filename:
        voxel_name_list.append(filename[0:-4])

with open(pos) as ad_in:
    for line in ad_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')
        tmp = ttmp[0].split('_')
        aa = tmp[0] + '_' + tmp[1]
        bb = tmp[2] + '_' + tmp[3]
        res1 = any(aa in voxel for voxel in voxel_name_list)
        res2 = any(bb in voxel for voxel in voxel_name_list)
        if res1 and res2:
            list_pos.append(ttmp[0])
        else:
            print aa + '  ' + bb
    list_pos.sort()
    list_pos = list(set(list_pos))
    print list_pos
ad_in.close()

with open(neg) as ot_in:
    for line in ot_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')
        tmp = ttmp[0].split('_')
        aa = tmp[0] + '_' + tmp[1]
        bb = tmp[2] + '_' + tmp[3]
        res1 = any(aa in voxel for voxel in voxel_name_list)
        res2 = any(bb in voxel for voxel in voxel_name_list)
        if res1 and res2:
            list_neg.append(ttmp[0])
        else:
            print aa + '  ' + bb

    list_neg.sort()
    list_neg = list(set(list_neg))
ot_in.close()

if os.path.exists("pos-new-ATP.lst"):
    os.remove("pos-new-ATP.lst")
with open("pos-new-ATP.lst", "w") as outf:
    for i in range(len(list_pos)):
        outf.write('%s\n' % list_pos[i])
outf.close()

if os.path.exists("neg-new-ATP.lst"):
    os.remove("neg-new-ATP.lst")
with open("neg-new-ATP.lst", "w") as outf:
    for i in range(len(list_neg)):
        outf.write('%s\n' % list_neg[i])
outf.close()
