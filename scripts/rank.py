import os
import random
import numpy as np
import utils
import torchfile
import pickle
import sys
sys.path.append("..")
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

random.seed(opts.seed)
type_embedding = opts.embtype 
# 'image'
# type_embedding = 'recipe'
print(opts.path_results)
with open(os.path.join(opts.path_results,'img_embeds.pkl'),'rb') as f:
    im_vecs = pickle.load(f)
with open(os.path.join(opts.path_results,'rec_embeds.pkl'),'rb') as f:
    instr_vecs = pickle.load(f)
with open(os.path.join(opts.path_results,'rec_ids.pkl'),'rb') as f:
    names = pickle.load(f)

# Sort based on names to always pick same samples for medr
idxs = np.argsort(names)
names = names[idxs]

# Ranker
N = opts.medr
idxs = range(N)

glob_rank = []
glob_recall = {1:0.0,5:0.0,10:0.0}
for i in range(10):

    ids = random.sample(range(0,len(names)), N)
    im_sub = im_vecs[ids,:]
    instr_sub = instr_vecs[ids,:]
    ids_sub = names[ids]

    # if params.embedding == 'image':
    if type_embedding == 'image':
        sims = np.dot(im_sub,instr_sub.T) # for im2recipe
    else:
        sims = np.dot(instr_sub,im_sub.T) # for recipe2im

    med_rank = []
    recall = {1:0.0,5:0.0,10:0.0}

    for ii in idxs:

        name = ids_sub[ii]
        # get a column of similarities
        sim = sims[ii,:]

        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()

        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)

        if (pos+1) == 1:
            recall[1]+=1
        if (pos+1) <=5:
            recall[5]+=1
        if (pos+1)<=10:
            recall[10]+=1

        # store the position
        med_rank.append(pos+1)

    for i in recall.keys():
        recall[i]=recall[i]/N

    med = np.median(med_rank)
    print("median", med)

    for i in recall.keys():
        glob_recall[i]+=recall[i]
    glob_rank.append(med)

for i in glob_recall.keys():
    glob_recall[i] = glob_recall[i]/10
print("Mean median", np.average(glob_rank))
print("Recall", glob_recall)
