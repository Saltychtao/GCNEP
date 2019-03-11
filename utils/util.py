import numpy as np
import torch


def pad(data,pad_idx,max_len=None):
    if max_len is None:
        max_len = max([len(instance) for instance in data])
    return [instance + [pad_idx]*max((max_len-len(instance),0)) for instance in data]


def one_hot_to_labels(tensor,one_label=None):
    # tensor bsize * N_C
    bsize,_ = tensor.size()
    if one_label is not None:
        labels = [[label.item()] for label in one_label]
    else:
        labels = [[] for i in range(bsize)]
    nonzeros = tensor.nonzero()
    for idx_label in nonzeros:
        idx = idx_label[0]
        label = idx_label[1].item()
        labels[idx].append(label)
    return labels

def load_pretrained(filepath,vocab,dim,device,pad_idx):

    vecs = np.random.normal(0,1,(len(vocab),dim))
    vecs[pad_idx] = np.zeros((dim))
    cnt = 0
    with open(filepath,'r') as f:
        f.readline()
        for line in f:
            splited = line.split(' ')
            word = splited[0]
            if word not in vocab:
                continue
            cnt += 1
            vec = [float(f) for f in splited[1:]]
            vecs[vocab[word]] = vec
        print('Found word vectors: {}/{}'.format(cnt,len(vocab)))
    tensor= torch.from_numpy(vecs)
    return tensor.float().to(device)
