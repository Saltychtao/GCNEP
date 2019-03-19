import torch
from torch.utils.data import DataLoader
from dataloader.simpleQA_dataloader import SimpleQADataset
from learner.SimpleQA import SimpleQA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    import time
    start_time = time.time()
    vocab = SimpleQADataset.load_vocab(args)
    args.kb_triplets = torch.load(args.kb_triplets_pth)
    train_dataset,dev_dataset,test_dataset = SimpleQADataset.load_dataset(args)
    end_time = time.time()
    print('Loaded dataset in {:.2f}s'.format(end_time - start_time))

    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.all_relation_words = vocab.get_all_relation_words()
    args.n_entities = len(vocab.etoi)

    if args.word_pretrained_pth is not None:
        args.word_pretrained = torch.load(args.word_pretrained_pth)
        print(' Pretrained word embedding loaded!')
    else:
        args.word_pretrained = None
        print(' Using random initialized word embedding.')

    if args.relation_pretrained_pth is not None:
        args.relation_pretrained = torch.load(args.relation_pretrained_pth)
        print(' Pretrained label word embedding loaded!')
    else:
        args.relation_pretrained = None
        print(' Using random initialized label word embedding.')

    train(args,train_dataset,dev_dataset,test_dataset,vocab,SimpleQADataset.collate_fn)


def train(args,train_dataset,dev_dataset,test_dataset,vocab,collate_fn):

    train_iter = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)
    dev_iter = DataLoader(dataset=dev_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)
    test_iter = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)

    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)

    args.padding_idx = 0

    print('Building Model...',end='')
    model = SimpleQA(args).to(device)
    print('Done')

    best_acc = -1.
    patience = args.patience
    test_acc = -1.
    for epoch in range(args.epoch):
        model.train_epoch(train_iter)
        dev_acc = model.evaluate(dev_iter)
        patience -= 1

        print(' \nEpoch {}, Dev Acc : {:.2f}, Test Acc:{:.2f}'.format(epoch,dev_acc*100,test_acc*100))
        if patience > 0 and dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(),args.save_pth)
            patience = args.patience

        if patience == 0:
            model = SimpleQA(args).to(device)
            model.load_state_dict(torch.load(args.save_pth))

            dev_acc = model.evaluate(dev_iter)
            test_acc = model.evaluate(test_iter)
            print('Dev Acc: {:.2f}, Test Acc :{:.2f}'.format(dev_acc*100,test_acc*100))
            return test_acc


class DefaultConfig:
    def __init__(self):
        self.word_dim = 300
        self.hidden_dim = 150
        self.relation_dim = 300
        self.batch_size = 64
        self.epoch = 100
        self.data_dir = 'data/SimpleQA'
        self.lr = 1e-3
        self.margin = 0.1
        self.ns = 256
        self.patience = 5
        self.freeze = True
        self.num_bases = 100
        self.num_hidden_layers = 4
        self.rgcn_dropout = 0.0

        self.padding_idx = 0
        self.pad_token = '<pad>'
        self.unk_idx = 1
        self.unk_token = '<unk>'
        self.relation_file = 'data/SimpleQA/FB2M.rel_voc.pickle'
        self.vocab_pth = 'data/SimpleQA/vocab.pth'

        self.train_dataset_pth = './data/SimpleQA/train.pkl'
        self.dev_dataset_pth = './data/SimpleQA/dev.pkl'
        self.test_dataset_pth = './data/SimpleQA/test.pkl'
        self.graph_file = './data/Freebase-2M/FB2M_subgraph.txt'

        self.save_pth = 'results/simpleQA/model_layer6.pth'

        self.word_pretrained_pth = './data/SimpleQA/word_pretrained.pth'
        # self.word_pretrained_pth = None
        self.graph_pth = './data/SimpleQA/subgraph.pth'
        self.kb_triplets_pth = './data/SimpleQA/kb_triplets.pth'
        self.relation_pretrained_pth = None

        self.glove_pth = '/home/user_data/lijh/data/english_embeddings/glove.6B.300d.txt'


class TestConfig:
    def __init__(self):
        self.word_dim = 3
        self.hidden_dim = 3
        self.relation_dim = 6
        self.batch_size = 10
        self.epoch = 5
        self.data_dir = 'data/SimpleQA'
        self.lr = 1e-3
        self.ns = 4
        self.padding_idx = 0
        self.pad_token = '<pad>'
        self.unk_idx = 1
        self.unk_token = '<unk>'
        self.relation_file = 'data/SimpleQA/FB2M.rel_voc.pickle'
        self.vocab_pth = 'data/SimpleQA/vocab.pth'

        self.train_dataset_pth = './data/SimpleQA/train.pkl'
        self.dev_dataset_pth = './data/SimpleQA/dev.pkl'
        self.test_dataset_pth = './data/SimpleQA/test.pkl'

        self.word_pretrained_path = '/home/user_data/lijh/data/glove.6B.300d.txt'
        self.relation_pretrained_path = None


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--train':
        args = DefaultConfig()
        main(args)
    elif sys.argv[1] == '--test':
        args = TestConfig()
        main(args)
    elif sys.argv[1] == '--generate':
        args = DefaultConfig()
        SimpleQADataset.generate_dataset(args)
        # SimpleQADataset.generate_embedding(args,device)
        SimpleQADataset.generate_graph(args,device)
