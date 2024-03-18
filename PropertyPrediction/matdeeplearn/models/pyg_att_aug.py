"""Implementation based on the template of ALIGNN."""
import math
from typing import Tuple, Optional, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from typing import Literal
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch_geometric

class MatformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MatformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                   bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.bn = nn.BatchNorm1d(out_channels * heads)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out)) # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3) 
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels))) 
        out = self.msg_layer(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
class Matformer_AUG(torch.nn.Module):
    def __init__(
            self,
            data,
            conv_layers: int = 5,
            node_features: int = 128,
            fc_layers: int = 1,
            fc_features: int = 512,
            node_layer_head: int = 2,
            link: Literal["identity", "log", "logit"] = "identity",
            zero_inflated: bool = False,
            pool="global_mean_pool",
            batch_norm="True",
            batch_track_stats="True",
            act="silu",
            dropout_rate=0.0,
            **kwargs
    ):
        
        super().__init__()

        if batch_track_stats == "True":
            batch_track_stats = True
        else:
            batch_track_stats = False

        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.dropout_rate = dropout_rate

        self.node_input_features = data.num_features
        self.edge_input_features = data.edge_attr.shape[1]

        print("node_input_features", self.node_input_features)
        print("edge_input_features", self.edge_input_features)

        self.conv_layers = conv_layers
        self.node_features = node_features
        self.fc_layers = fc_layers
        self.fc_features = fc_features
        self.node_layer_head = node_layer_head

        self.atom_embedding = nn.Linear(
            self.node_input_features, self.node_features
        )

        self.edge_embedding = nn.Linear(
            self.edge_input_features, self.node_features
        )
        
        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=self.node_features, out_channels=self.node_features, heads=self.node_layer_head, edge_dim=self.node_features)
                for _ in range(self.conv_layers)
            ]
        )

        self.fc_linear_layers = nn.ModuleList()

        self.fc_linear_layers.append(nn.Linear(self.node_features+32+7, self.fc_features))

        for _ in range(self.fc_layers - 1):
            self.fc_linear_layers.append(nn.Linear(self.fc_features, self.fc_features))

        self.sigmoid = nn.Sigmoid()

        self.fc_out = nn.Linear(self.fc_features, 1)

        # self.link = None
        # self.link_name = link
        # if link == "identity":
        #     self.link = lambda x: x
        # elif link == "log":
        #     self.link = torch.exp
        #     avg_gap = 0.7
        #     if not self.zero_inflated:
        #         self.fc_out.bias.data = torch.tensor(
        #             np.log(avg_gap), dtype=torch.float
        #         )
        # elif link == "logit":
        #     self.link = torch.sigmoid

    def forward(self, data) -> torch.Tensor:

        node_features = self.atom_embedding(data.x)
        edge_features = self.edge_embedding(data.edge_attr)

        for att_layer in self.att_layers:
            node_features = att_layer(node_features, data.edge_index, edge_features)

        features = getattr(torch_geometric.nn, self.pool)(node_features, data.batch)

        features = torch.cat([features, data.pge], dim=1)
        features = torch.cat([features, data.cse], dim=1)

        for fc_layer in self.fc_linear_layers:
            features = getattr(F, self.act)(fc_layer(features))
            features = F.dropout(features, p=self.dropout_rate, training=self.training)
            
        out = self.fc_out(features)
        # if self.link is not None:
        #     out = self.link(out)

        return out