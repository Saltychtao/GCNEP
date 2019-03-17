import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LSTMEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 batch_first,
                 bidirectional):
        super(LSTMEncoder, self).__init__()
        input_size = input_size
        dropout = 0 if num_layers == 1 else dropout
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                           num_layers=num_layers,dropout=dropout,batch_first=batch_first,
                           bidirectional=bidirectional)

    def forward(self,inputs,lengths,need_sort=False):
        if need_sort:
            lengths,perm_idx = lengths.sort(0,descending=True)
            inputs = inputs[perm_idx]
        bsize = inputs.size()[0]
        # state_shape = self.config.n_cells,bsize,self.config.d_hidden
        # h0 = c0 = inputs.new_zeros(state_shape)
        inputs = pack(inputs,lengths,batch_first=True)
        outputs,(ht,ct) = self.rnn(inputs)
        outputs,_ = unpack(outputs,batch_first=True)

        if need_sort:
            _,unperm_idx = perm_idx.sort(0)
            outputs = outputs[unperm_idx]
        return outputs,ht.permute(1,0,2).contiguous().view(bsize,-1)


def mean_pool(input,length):
    # input: bsize *  seq_len * dim
    # length: bsize
    length = length.unsqueeze(1)
    input = torch.sum(input,1).squeeze()
    return torch.div(input,length.float())


def max_pool(input):
    input[input == 0] = -1e9
    return torch.max(input,dim=1)[0]


class GateNetwork(nn.Module):
    def __init__(self,hidden_dim):
        super(GateNetwork, self).__init__()
        self.gate_fc1 = nn.Linear(hidden_dim,hidden_dim)
        self.gate_fc2 = nn.Linear(hidden_dim,hidden_dim)

    def forward(self,input1,input2):
        assert input1.size() == input2.size()
        gate = torch.sigmoid(
            self.gate_fc1(input1) +
            self.gate_fc2(input2)
        )
        return torch.mul(gate,input2)
