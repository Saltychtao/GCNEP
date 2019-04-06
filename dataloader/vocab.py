from utils.util import pad
import numpy as np
class Vocab(object):
    def __init__(self):
        pass

    def renew_vocab(self,data,name):
        for d in data:
            if d in getattr(self,name):
                continue
            else:
                getattr(self,name)[d] = len(getattr(self,name))


class SimpleQAVocab(Vocab):

    def __init__(self):
        self.relIdx2wordIdx = {}
        self.relIdx2nameIdx = {}

    def get_all_relation_words(self):
        n_relations = len(self.rtoi)
        max_len = 0
        for r,idx in self.rtoi.items():
            max_len = max(max_len,len(self.relIdx2wordIdx[idx]))
        print('Max length of relation is {}'.format(max_len))
        return np.array(pad([self.relIdx2wordIdx[i] for i in range(n_relations)],0,max_len=max_len))

    def get_all_relation_names(self):
        n_relations = len(self.rtoi)
        return np.array(pad([self.relIdx2nameIdx[i] for i in range(n_relations)],0,max_len=10))
