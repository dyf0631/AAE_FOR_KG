# import scipy.sparse as sp
# import pandas as pd
from tqdm import tqdm
import pickle
from collections import defaultdict, OrderedDict
import numpy as np
import os


def pk_save(obj, file_path):
    return pickle.dump(obj, open(file_path, 'wb'))


def pk_load(file_path):
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    else:
        return None


# output_files
x_output_file = 'ind.decagon.allx'
graph_output_file = 'ind.decagon.graph'
# output graph
multi_graph = {}

voc = OrderedDict()
multi_graph_i = pk_load(graph_output_file)
for type, dict_ls in multi_graph_i.items():
    if type not in multi_graph.keys():
        multi_graph[type] = defaultdict(list)
    for drug_k, drug_ls in dict_ls.items():
        if drug_k not in voc.keys():
            voc[drug_k] = len(voc)
        for drug_v in drug_ls:
            if drug_v not in voc.keys():
                voc[drug_v] = len(voc)
            multi_graph[type][voc[drug_k]].append(voc[drug_v])


triplet = []
entity2idfile = open("entity2id.txt",'w')
rel2idfile = open("rel2id.txt",'w')

entityset = set()

typekeys = multi_graph.keys()
for typekey in typekeys:
    typelist = multi_graph[typekey]
    rel2idfile.write(str(typekey)+"\t"+str(typekey)+"\n")
    keys = typelist.keys()
    for i in keys:
        if i not in entityset:
            entityset.add(i)
        for ls in typelist[i]:
            triplet.append(str(i)+"\t"+str(typekey)+"\t"+str(ls)+"\n")

import random

random.shuffle(triplet)
train = open("train.txt",'w')
test = open("test.txt",'w')
valid = open("valid.txt",'w')

spt = len(triplet)*0.8
spt = int(spt)
for i in range(spt):
	train.write(triplet[i])

spt1 = len(triplet)*0.9
spt1 = int(spt1)

for i in range(spt,spt1):
	test.write(triplet[i])

for i in range(spt1,len(triplet)):
	valid.write(triplet[i])

train.close()
test.close()
valid.close()



for i in entityset:
    entity2idfile.write(str(i)+"\t"+str(i)+"\n")




# # save graph and node feature
# pk_save(multi_graph, graph_output_file)

# n_drugs = len(voc)
# drug_feat = sp.identity(n_drugs)
# pk_save(drug_feat, x_output_file)
