import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Sigmoid, Module, ModuleList, Parameter, LayerNorm
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv
from torch_geometric.utils import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

class GGCNConv(Module):
    def __init__(self, dim, act, dropout_rate, batch_track_stats, **kwargs):
        super(GGCNConv, self).__init__()

        self.act = act
        self.dropout_rate = dropout_rate

        self.gate_linear=Linear(3*dim, dim)

        self.src_linear=Linear(dim, dim)
        self.dst_linear=Linear(dim, dim)

        # self.node_bn=DiffGroupNorm(dim, 10, track_running_stats=batch_track_stats)
        # self.edge_bn=DiffGroupNorm(dim, 10, track_running_stats=batch_track_stats),
        self.node_bn=BatchNorm1d(dim, track_running_stats=batch_track_stats)
        self.edge_bn=BatchNorm1d(dim, track_running_stats=batch_track_stats)

    def forward(self, x, edge_index, edge_attr):
        # m=C*x_i+D*x_j+E*e_ij
        m=self.gate_linear(torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1))

        # h=A*x+sum(B*x_j*eta)
        # eta=sigma(m_j)/sigma(m_k)
        sigma=torch.sigmoid(m/pow(m.shape[-1], 0.5))
        src=self.src_linear(x)
        dst=self.dst_linear(x[edge_index[1]])
        # eta=sigma*dst/(scatter(sigma, edge_index[0], dim=0, reduce='sum')[edge_index[0]]+1e-8)
        h=src+scatter(sigma*dst, edge_index[0], dim=0, reduce='sum')

        return self.act(self.node_bn(h)), self.act(self.edge_bn(m))
        # return self.act(self.node_bn(h))

class GGCN_EDGE(Module):
    def __init__(
        self,
        data,
        dim1=50,
        dim2=100,
        pre_fc_count=1,
        gc_count=5,
        post_fc_count=1,
        pool='global_add_pool',
        pool_order='early',
        batch_norm='True',
        batch_track_stats='True',
        act='relu',
        dropout_rate=0.0,
        **kwargs
    ):
        super(GGCN_EDGE, self).__init__()

        if batch_track_stats=='True':
            self.batch_track_stats=True
        else:
            self.batch_track_stats=False

        self.batch_norm=batch_norm
        self.pool=pool
        self.act=act
        self.pool_order=pool_order
        self.dropout_rate=dropout_rate

        # Determine gc dimension
        assert gc_count > 0, "Need at least one GGCN layer"
        if pre_fc_count == 0:
            gc_dim=data.num_features
        else:
            gc_dim=dim1

        # Determine post_fc dimension
        if post_fc_count == 0:
            post_fc_dim=data.num_features
        else:
            post_fc_dim=dim1
        
        # Determine output dimension length
        if data[0].y.ndim==0:
            output_dim=1
        else:
            output_dim=len(data[0].y)

        # Set up pre-GNN dense layers
        if pre_fc_count > 0:
            self.pre_lin_list_E=ModuleList()
            self.pre_lin_list_N=ModuleList()

            for i in range(pre_fc_count):
                if i==0:
                    self.pre_lin_list_N.append(Linear(data.num_features, dim1))
                    self.pre_lin_list_E.append(Linear(data.num_edge_features, dim1))
                else:
                    self.pre_lin_list_N.append(Linear(dim1, dim1))
                    self.pre_lin_list_E.append(Linear(dim1, dim1))
        elif pre_fc_count==0:
            self.pre_lin_list_E=ModuleList()
            self.pre_lin_list_N=ModuleList()

        # Set up GNN layers
        self.conv_list=ModuleList()
        self.bn_list_nodes=ModuleList()
        self.bn_list_edges=ModuleList()
        for i in range(gc_count):
            self.conv_list.append(GGCNConv(dim1, getattr(F, act), dropout_rate, self.batch_track_stats))
            if self.batch_norm=='True':
                # self.bn_list_nodes.append(DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats))
                # self.bn_list_edges.append(DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)),
                self.bn_list_nodes.append(BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats))
                self.bn_list_edges.append(BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats))

        if post_fc_count > 0:
            self.post_lin_list = ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    lin = Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = Linear(dim2, output_dim)
        elif post_fc_count==0:
            self.post_lin_list = ModuleList()
            self.lin_out = Linear(post_fc_dim, output_dim)
    
    def forward(self, data):

        # Pre-GNN dense layers
        for i in range(len(self.pre_lin_list_N)):
            if i==0:
                out_x=self.pre_lin_list_N[i](data.x)
                out_x=getattr(F, self.act)(out_x)
                out_e=self.pre_lin_list_E[i](data.edge_attr)
                out_e=getattr(F, self.act)(out_e)
            else:
                out_x=self.pre_lin_list_N[i](out_x)
                out_x=getattr(F, self.act)(out_x)
                out_e=self.pre_lin_list_E[i](out_e)
                out_e=getattr(F, self.act)(out_e)
        prev_out_x=out_x
        prev_out_e=out_e

        # GNN layers
        for i in range(len(self.conv_list)):
            if len(self.pre_lin_list_N)==0 and i==0:
                if self.batch_norm=='True':
                    out_x, out_e=self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    # out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    out_x=self.bn_list_nodes[i](out_x)
                    out_e=self.bn_list_edges[i](out_e)
                else:
                    out_x, out_e=self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    # out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm=='True':
                    out_x, out_e=self.conv_list[i](out_x, data.edge_index, out_e)
                    # out_x = self.conv_list[i](out_x, data.edge_index, out_e)
                    out_x=self.bn_list_nodes[i](out_x)
                    out_e=self.bn_list_edges[i](out_e)
                else:
                    out_x, out_e=self.conv_list[i](out_x, data.edge_index, out_e)
                    # out_x = self.conv_list[i](out_x, data.edge_index, out_e)
            out_x=torch.add(out_x, prev_out_x)
            out_e=torch.add(out_e, prev_out_e)
            out_x=F.dropout(out_x, p=self.dropout_rate, training=self.training)
            out_e=F.dropout(out_e, p=self.dropout_rate, training=self.training)
            prev_out_x=out_x
            prev_out_e=out_e

        out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)

        for i in range(0, len(self.post_lin_list)):
            out_x = self.post_lin_list[i](out_x)
            out_x = getattr(F, self.act)(out_x)

        out = self.lin_out(out_x)
        
        return out