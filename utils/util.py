import numpy as np
import torch
import yaml


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
    tensor = torch.from_numpy(vecs)
    return tensor.float().to(device)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(open(args.config_file))
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                arg_dict[key] = []
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args
