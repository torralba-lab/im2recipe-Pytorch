# !/usr/bin/env python
import random
import pickle
import numpy as np
from proc import *
from tqdm import *
import torchfile
import time
import utils
import os
# from ..args import get_parser
import time
import lmdb
import shutil
import sys
sys.path.append("..")
from args import get_parser

# Maxim number of images we want to use per recipe
maxNumImgs = 5

def get_st(file):
    info = torchfile.load(file)

    ids = info[b'ids']

    imids = []
    for i,id in enumerate(ids):
        imids.append(''.join(chr(i) for i in id))

    st_vecs = {}
    st_vecs['encs'] = info['encs']
    st_vecs['rlens'] = info['rlens']
    st_vecs['rbps'] = info['rbps']
    st_vecs['ids'] = imids

    print(np.shape(st_vecs['encs']),len(st_vecs['rlens']),len(st_vecs['rbps']),len(st_vecs['ids']))
    return st_vecs

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

DATASET = opts.dataset

# don't use this file once dataset is clean
with open('remove1M.txt','r') as f:
    remove_ids = {w.rstrip(): i for i, w in enumerate(f)}

t = time.time()
print ("Loading skip-thought vectors...")

st_vecs_train = get_st(os.path.join(opts.sthdir, 'encs_train_1024.t7'))
st_vecs_val = get_st(os.path.join(opts.sthdir, 'encs_val_1024.t7'))
st_vecs_test = get_st(os.path.join(opts.sthdir, 'encs_test_1024.t7'))

st_vecs = {'train':st_vecs_train,'val':st_vecs_val,'test':st_vecs_test}
stid2idx = {'train':{},'val':{},'test':{}}

for part in ['train','val','test']:
    for i,id in enumerate(st_vecs[part]['ids']):
        stid2idx[part][id] = i

print ("Done.",time.time() - t)

print('Loading dataset.')
# print DATASET
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],DATASET)
print('Loading ingr vocab.')
with open(opts.vocab) as f_vocab:
    ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
    ingr_vocab['</i>'] = 1

with open('classes1M.pkl','rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)

st_ptr = 0
numfailed = 0

if os.path.isdir('../data/train_lmdb'):
    shutil.rmtree('../data/train_lmdb')
if os.path.isdir('../data/val_lmdb'):
    shutil.rmtree('../data/val_lmdb')
if os.path.isdir('../data/test_lmdb'):
    shutil.rmtree('../data/test_lmdb')

env = {'train' : [], 'val':[], 'test':[]}
env['train'] = lmdb.open('../data/train_lmdb',map_size=int(1e11))
env['val']   = lmdb.open('../data/val_lmdb',map_size=int(1e11))
env['test']  = lmdb.open('../data/test_lmdb',map_size=int(1e11))

print('Assembling dataset.')
img_ids = dict()
keys = {'train' : [], 'val':[], 'test':[]}
for i,entry in tqdm(enumerate(dataset)):

    ninstrs = len(entry['instructions'])
    ingr_detections = detect_ingrs(entry, ingr_vocab)
    ningrs = len(ingr_detections)
    imgs = entry.get('images')

    if ninstrs >= opts.maxlen or ningrs >= opts.maxlen or ningrs == 0 or not imgs or remove_ids.get(entry['id']):
        continue

    ingr_vec = np.zeros((opts.maxlen), dtype='uint16')
    ingr_vec[:ningrs] = ingr_detections 

    partition = entry['partition']

    stpos = stid2idx[partition][entry['id']] #select the sample corresponding to the index in the skip-thoughts data
    beg = st_vecs[partition]['rbps'][stpos] - 1 # minus 1 because it was saved in lua
    end = beg + st_vecs[partition]['rlens'][stpos]

    serialized_sample = pickle.dumps( {'ingrs':ingr_vec, 'intrs':st_vecs[partition]['encs'][beg:end],
        'classes':class_dict[entry['id']]+1, 'imgs':imgs[:maxNumImgs]} ) 

    with env[partition].begin(write=True) as txn:
        txn.put('{}'.format(entry['id']).encode('latin1'), serialized_sample)
    # keys to be saved in a pickle file    
    keys[partition].append(entry['id'])

for k in keys.keys():
    with open('../data/{}_keys.pkl'.format(k),'wb') as f:
        pickle.dump(keys[k],f)

print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))

