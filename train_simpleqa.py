import torch
from torch.utils.data import DataLoader
from dataloader.simpleQA_dataloader import SimpleQADataset
from learner.SimpleQA import SimpleQA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_folds(labelset,K=10):
    length = len(labelset)
    len_of_each_folds = length // K
    label_list = list(labelset)
    folds = []
    for i in range(0,length,len_of_each_folds):
        folds.append(label_list[i:min(i + len_of_each_folds,length)])
    return folds


def main(args):

    import time
    start_time = time.time()
    vocab = SimpleQADataset.load_vocab(args)
    train_dataset,dev_dataset,test_dataset = SimpleQADataset.load_dataset(args)
    end_time = time.time()
    print('Loaded dataset in {:.2f}s'.format(end_time - start_time))

    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.all_relation_words = vocab.get_all_relation_words()
    args.all_relation_names = vocab.get_all_relation_names()

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

    if args.mode == 'supervised':
        train(args,train_dataset,dev_dataset,test_dataset,vocab,SimpleQADataset.collate_fn)
    elif args.mode == 'zero-shot':
        train_rels = train_dataset.get_label_set()
        test_rels = test_dataset.get_label_set()
        folds = generate_folds(test_rels)

        for i,fold in enumerate(folds):
            print('Training Folds :{}'.format(i))
            train_subset = train_dataset.get_subset(fold,mode='unseen')
            dev_subset = dev_dataset.get_subset(fold,mode='unseen')
            test_subset = test_dataset.get_subset(fold,mode='seen')

            with open('fold/fold{}.txt'.format(str(i)),'w') as f:
                f.write('Training Subset labels: {}'.format(','.join(list(train_rels - set(fold)))) + '\n')
                f.write('Test Subset labels: {}\n'.format(','.join(fold)))

            test_acc = train(args,train_subset,dev_subset,test_subset,vocab,SimpleQADataset.collate_fn)
            with open('fold/fold{}.txt'.format(str(i)),'a') as f:
                f.write('Test Acc : {:.2f}'.format(test_acc))


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
        self.data_dir = 'data/SimpleQuestions_yu'
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
        self.relation_file = 'data/SimpleQuestions_yu/relation.2M.list'
        self.vocab_pth = 'data/SimpleQuestions_yu/vocab.pth'

        self.train_dataset_pth = './data/SimpleQuestions_yu/train.pkl'
        self.dev_dataset_pth = './data/SimpleQuestions_yu/dev.pkl'
        self.test_dataset_pth = './data/SimpleQuestions_yu/test.pkl'
        self.graph_file = './data/SimpleQuestions_yu/FB2M_subgraph.txt'

        self.save_pth = 'results/simpleQA/baseline.pth'

        self.word_pretrained_pth = './data/SimpleQuestions_yu/word_pretrained.pth'
        # self.word_pretrained_pth = None
        self.kb_triplets_pth = './data/SimpleQuestions_yu/kb_triplets.pth'
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
    if sys.argv[1] == '--supervised':
        args = DefaultConfig()
        args.mode = 'supervised'
        main(args)
    elif sys.argv[1] == '--test':
        args = TestConfig()
        main(args)
    elif sys.argv[1] == '--generate':
        args = DefaultConfig()
        SimpleQADataset.generate_dataset(args)
        SimpleQADataset.generate_embedding(args,device)
        # SimpleQADataset.generate_graph(args,device)
