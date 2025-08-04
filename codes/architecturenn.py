import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp

class AtomMessagePassing(MessagePassing):
    def __init__(self, in_edge_dim, in_node_dim, hidden_dim, out_dim, T=4, aggr='add'):
        super().__init__(aggr=aggr)
        self.T = T
        self.W_i = nn.Linear(in_node_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim + in_edge_dim, hidden_dim)
        self.W_o = nn.Linear(in_node_dim + hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        h0 = self.act(self.W_i(x))
        h = h0
        for t in range(1, self.T):
            m = self.propagate(edge_index, x=h, edge_attr=edge_attr)
            h = self.act(h0 + self.W_h(m))
        m_T = self.propagate(edge_index, x=h, edge_attr=None)
        h_T = self.act(self.W_o(torch.cat([x, m_T], dim=-1)))
        return h_T

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return x_j
        else:
            return torch.cat([x_j, edge_attr], dim=-1)

class GCN(torch.nn.Module):
    def __init__(self, conv_function, input_channels, input_channels2, embedding_size, hidden_dim, linear_size, add_params_num=0):
        super(GCN, self).__init__()
        self.crafted_add_params_num = add_params_num

        self.conv1a = conv_function(1, input_channels, hidden_dim, embedding_size)
        self.conv1b = conv_function(1, input_channels2, hidden_dim, embedding_size)
        self.dropout1 = torch.nn.Dropout(0.2)

        self.linear1 = Linear(2*embedding_size+add_params_num, linear_size[0])
        self.linear2 = Linear(linear_size[0],linear_size[1])
        self.dropout2 = torch.nn.Dropout(0.3)
        self.bnf = torch.nn.BatchNorm1d(linear_size[-1])
        self.out = Linear(linear_size[-1], 1)


    def forward(self, x_l, edge_index_l, edge_weight_l, x_s, edge_index_s, edge_weight_s, batch_index_l, batch_index_s, cond=None):
        hidden1 = self.conv1a(x_l, edge_index_l, edge_weight_l).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = gap(hidden1, batch_index_l)

        hidden2 = self.conv1b(x_s, edge_index_s, edge_weight_s).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = gap(hidden2, batch_index_s)

        batch_index = torch.cat([batch_index_l, batch_index_s], dim=0)

        if self.crafted_add_params_num != 0:
            cond = cond.unsqueeze(1)
            hidden = torch.cat([hidden1, hidden2, cond], dim=1)
        else:
            hidden = torch.cat([hidden1, hidden2], dim=1)

        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.bnf(hidden)
        hidden = torch.nn.functional.relu(hidden)
        out = self.out(hidden)

        return out, hidden