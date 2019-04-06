import torch
import torch.nn as nn
from sklearn import metrics

from utils.module import LSTMEncoder,mean_pool,max_pool,GateNetwork
from utils.metric import micro_precision,macro_precision
from model.GCN import RGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_step = 0


class SimpleQA(nn.Module):
    def __init__(self,args):
        super(SimpleQA, self).__init__()

        if args.word_pretrained is None:
            self.word_embedding = nn.Embedding(args.n_words,args.word_dim,args.padding_idx)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(args.word_pretrained,freeze=args.freeze)

        if args.use_gcn:
            self.gcns = nn.ModuleList()
            for g in args.relation_graphs:
                gcn = RGCN(
                    g,
                    args.n_relations,
                    args.sub_relation_dim,
                    args.sub_relation_dim,
                    args.relation_pretrained,
                    args.num_hidden_layers,
                    args.rgcn_dropout,
                    use_cuda=True
                )
                self.gcns.append(gcn)
        else:
            if args.relation_pretrained is None:
                self.relation_embedding = nn.Embedding(args.n_relations,args.relation_dim,args.padding_idx)
            else:
                self.relation_embedding = nn.Embedding.from_pretrained(args.relation_pretrained,freeze=False)

        self.word_encoder = LSTMEncoder(
            input_size=args.word_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.question_encoder = LSTMEncoder(
            input_size=2*args.hidden_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.gate = GateNetwork(2*args.hidden_dim)

        self.loss_fn = nn.MultiMarginLoss(margin=args.margin)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=args.lr)

        self.ns = args.ns
        self.score_function = nn.CosineSimilarity(dim=2)

        self.all_relation_words = args.all_relation_words

        self.n_relations = args.n_relations
        self.args = args

        global global_step
        global_step = 0

    def get_relation_embedding(self):
        if self.args.use_gcn:
            relation_embedding = []
            for gcn in self.gcns:
                embed = gcn.forward()
                relation_embedding.append(embed)
            if self.args.graph_aggr == 'concat':
                return torch.cat(relation_embedding,dim=1)
            elif self.args.graph_aggr == 'mean':
                return torch.stack(relation_embedding,dim=0).mean(0)
            elif self.args.graph_aggr == 'max':
                return torch.stack(relation_embedding,dim=0).max(0)[0]
        else:
            all_relations = torch.tensor([i for i in range(self.n_relations)]).to(device)
            return self.relation_embedding(all_relations)

    def get_shift(self):
        with torch.no_grad():
            if not self.args.use_gcn:
                relation_embedding = self.get_relation_embedding()
            else:
                relation_embedding = self.get_relation_embedding()
            dim = relation_embedding.size()[1]
            seen_relation = relation_embedding[self.args.seen_idx,:]
            unseen_relation = relation_embedding[self.args.unseen_idx,:]
            seen_relation_center = seen_relation.mean(dim=0)
            unseen_relation_center = unseen_relation.mean(dim=0)
            return torch.norm(seen_relation_center-unseen_relation_center,p=2).item() / dim

    def forward(self,question,relation):

        n_rels = relation.size()[1]
        question_length = (question != self.args.padding_idx).sum(dim=1).long().to(device)
        question_mask = (question != self.args.padding_idx)
        question = self.word_embedding(question)
        low_question_repre = self.word_encoder(question,question_length,need_sort=True)[0]

        high_question_repre = self.question_encoder(low_question_repre,question_length,need_sort=True)[0]
        question_repre = (low_question_repre + high_question_repre)  # bsize * seq_len * (2*hidden)
        question_repre = max_pool(question_repre,question_mask) # bsize * (2*hidden)

        # single relation repre
        single_relation_repre = self.get_relation_embedding().unsqueeze(1)
        single_relation_repre = self.word_encoder(single_relation_repre,torch.tensor([1]*self.n_relations),need_sort=True)[0] # bsize * 1 * (2*hidden)

        # relation words repre
        all_relation_words = torch.tensor(self.all_relation_words).to(device)  # n_relations * max_len

        relation_words_lengths = (all_relation_words != self.args.padding_idx).sum(dim=-1).long().to(device)
        relation_words_mask = (all_relation_words != self.args.padding_idx)
        relation_words_repre = self.word_embedding(all_relation_words)
        relation_words_repre = max_pool(self.word_encoder(relation_words_repre,relation_words_lengths,need_sort=True)[0],relation_words_mask) # bsize * (2*hidden)

        # relation_repre = self.gate(single_relation_repre,relation_words_repre)
        relation_repre = torch.cat([relation_words_repre.unsqueeze(1),single_relation_repre],dim=1).max(dim=1)[0]

        relation_repre = relation_repre[relation,:]  # bsize * n_rels * hidden

        scores = self.score_function(relation_repre,question_repre.unsqueeze(1).repeat(1,n_rels,1))

        return scores

    def train_epoch(self,train_iter):
        self.train()

        total_batch = len(train_iter)
        total = 0.
        loss = 0.0
        cur_batch = 1
        correct = 0

        for batch in train_iter:
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            scores = self.forward(question,relation)  # bsize * (1 + ns)
            batch_loss = self.loss_fn(scores,labels)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            cur_batch += 1

            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += bsize

            loss += batch_loss
            print('\r Batch {}/{}, Training Loss:{:.4f}, Training Acc:{:.2f}'.format(cur_batch,total_batch,loss/cur_batch,correct/total*100),end='')
            center_distance = self.get_shift()
            global global_step
            global_step += 1
            self.args.writer.add_scalar('center_distance',center_distance,global_step)

    def evaluate(self,dev_iter):
        self.eval()
        total = 0
        pred = []
        gold = []
        for batch in dev_iter:
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            gold.extend(relation[range(bsize),labels].tolist())

            relation_mask = (1e9*(relation != 0) -1e9).float() # 1 -> 0, 0 -> -1e9
            scores = self.forward(question,relation) # bsize * (1 + neg_num)
            correct_idx =  (scores+relation_mask).argmax(dim=1)
            pred.extend(relation[range(bsize),correct_idx].tolist())
            # correct += ((scores+relation_mask).argmax(dim=1) == labels).sum().item()
            total += bsize

        return micro_precision(pred,gold),macro_precision(pred,gold)
