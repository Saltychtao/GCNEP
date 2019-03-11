class Vocab(object):
    def __init__(self):
        pass

    def renew_vocab(self,data,name):
        for d in data:
            if d in getattr(self,name):
                continue
            else:
                getattr(self,name)[d] = len(getattr(self,name))
