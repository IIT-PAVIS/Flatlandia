"""
Name: models.py
Description: A set of baseline models for 3DoF visual localization on annotated maps.
-----
Author: Matteo Toso.
Licence: MIT. Copyright 2023, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""

import torch
import torch.nn as nn
from dgl.nn import EGATConv
import torch.nn.functional as F
import math
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from torch.nn.functional import normalize
import dgl.function as fn

#########################################################################################################
#  Baseline Models for the Fine-grained Localization Task                                               #
#########################################################################################################


def select_module(mode, angles_as_trig=0):
    if mode == "MLP":
        model = MLPModule(3)
    elif mode == "GAT+MLP":
        model = GATMLPModule(3, n_classes=42, n_layers=2, split=True, out_size=64,
                             only_query=True, edge_dim_in=2+angles_as_trig)
    elif mode == "MLP+ATT+MLP":
        model = MLPATTMLPModule(4, n_classes=42, n_layers=2, split=True, out_size=64)
    elif mode == "GAT+ATT+MLP":
        model = GATATTMLPModule(3, n_classes=42, n_layers=2, split=True, out_size=64,
                                only_query=True, edge_dim_in=2+angles_as_trig)
    else:
        print('Module choices are [MLP, GAT+MLP, MLP+ATT+MLP, GAT+ATT+MLP]')
        return 0
    return model


class MLPModule(torch.nn.Module):
    def __init__(self, num_heads, n_classes=42, n_layers=2, out_size=64, split=True):
        super(MLPModule, self).__init__()
        node_in_size = n_classes + 2
        self.query_proj = torch.nn.Linear(node_in_size, out_size)  # To fuse the edges
        self.ref_proj = torch.nn.Linear(node_in_size, out_size)  # To fuse the edges

        self.merge = torch.nn.Sequential(
            torch.nn.Linear(out_size * 2, out_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size * 2, out_size * 2),
            torch.nn.ReLU(),
        )

        self.regress = torch.nn.Sequential(
            torch.nn.Linear(out_size * 2, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, 3)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, g_all, all_node_data, all_edge_data, n_r):
        ref = all_node_data[0:n_r]
        query = all_node_data[n_r:]
        query_data = self.query_proj(query)
        ref_data = self.ref_proj(ref)

        # Pool the Query
        query_feat = torch.max(query_data, dim=0).values.unsqueeze(0)

        # Tile query over reference and merge
        query_feat = query_feat.tile((ref_data.size(0), 1))

        ref_query = torch.cat((ref_data, query_feat), dim=1)
        ref_query = self.merge(ref_query)

        feat = torch.max(ref_query, dim=0).values
        loc = self.regress(feat.unsqueeze(0))
        return None, None, loc


class GATMLPModule(torch.nn.Module):
    def __init__(self, num_heads, n_classes=42, n_layers=2, out_size=64, split=True, only_query=True, edge_dim_in=2):
        super(GATMLPModule, self).__init__()
        self.only_query = only_query
        self.n_layers = n_layers
        self.split = split
        node_in_size = n_classes + 2
        self.softmax = torch.nn.Softmax(dim=-1)
        self.edge_lin = torch.nn.Linear(edge_dim_in, out_size)
        self.node_proj = torch.nn.Linear(node_in_size, out_size)
        # We use two GAT layers
        self.conv1 = EGATConv(in_node_feats=out_size, out_node_feats=2 * out_size, num_heads=num_heads,
                              in_edge_feats=out_size, out_edge_feats=2 * out_size)
        self.conv2 = EGATConv(in_node_feats=2 * out_size * num_heads, out_node_feats=4 * out_size, num_heads=num_heads,
                              in_edge_feats=2 * out_size * num_heads, out_edge_feats=out_size)
        if self.split:
            self.ref_query_conv = EGATConv(in_node_feats=out_size, out_node_feats=2 * out_size, num_heads=num_heads,
                                           in_edge_feats=out_size, out_edge_feats=out_size * 2)

        # To run multiple times the network we need to fuse all the heads outputs into one
        self.lin1 = torch.nn.Linear(num_heads * out_size, out_size)  # To fuse the edges
        self.lin2 = torch.nn.Linear(num_heads * 4 * out_size, out_size)  # To fuse the nodes
        self.lin2a = torch.nn.Linear(num_heads * 2 * out_size, out_size)  # To fuse the nodes
        self.lin3 = torch.nn.Linear(num_heads * 2 * out_size, out_size)  # To fuse the nodes
        self.line_predict = torch.nn.Linear(out_size, 1)

        self.regress = torch.nn.Sequential(
            torch.nn.Linear(out_size, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, 3)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, g_all, all_node_data, all_edge_data, n_r):

        g, ref_and_query = g_all
        node_data0 = all_node_data[0:g.num_nodes()]
        edge_data = all_edge_data[0:g.num_edges()]

        node_data = self.node_proj(node_data0)
        edge_data = self.edge_lin(edge_data)
        edge_data_link = self.edge_lin(all_edge_data[g.num_edges():])

        for step in range(self.n_layers):
            h, e = self.conv1(g, node_data, edge_data)
            h = F.relu(h)
            e = F.relu(e)
            h, e = self.conv2(g, h.view(len(node_data), -1), e.view(len(edge_data), -1))
            edge_data = F.relu(self.lin1(e.view(len(edge_data), -1)))
            node_data = F.relu(self.lin2(h.view(len(node_data), -1)))

            # Perform the Query to Reference update
            h, e3 = self.ref_query_conv(ref_and_query, node_data, edge_data_link)
            edge_data_link = F.relu(self.lin3(e3.view(len(edge_data_link), -1)))
            node_data = node_data + F.relu(self.lin2a(h.view(len(node_data), -1)))


        # edge_data = self.line_predict(edge_data)
        if self.only_query:
            node_data = node_data[n_r:]

        feat = torch.max(node_data, dim=0).values
        loc = self.regress(feat.unsqueeze(0))
        return edge_data, node_data, loc


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MLPATTMLPModule(torch.nn.Module):
    def __init__(self, num_heads, n_classes=42, n_layers=2, out_size=64, split=True):
        super(MLPATTMLPModule, self).__init__()
        node_in_size = n_classes + 2
        self.query_proj = torch.nn.Linear(node_in_size, out_size)  # To fuse the edges
        self.ref_proj = torch.nn.Linear(node_in_size, out_size)  # To fuse the edges

        self.query_multihead_attn = MultiheadAttention(out_size, int(out_size), num_heads)
        self.query_ref_multihead_attn = MultiheadAttention(out_size * 2, int((out_size * 2)), num_heads)

        self.merge = torch.nn.Sequential(
            torch.nn.Linear(out_size * 2, out_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size * 2, out_size * 2),
            torch.nn.ReLU(),
        )

        self.regress = torch.nn.Sequential(
            torch.nn.Linear(out_size * 2, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, 3)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, g_all, all_node_data, all_edge_data, n_r):
        ref = all_node_data[0:n_r]
        query = all_node_data[n_r:]
        query_data = self.query_proj(query)
        ref_data = self.ref_proj(ref)

        # Pool the Query
        attn_out = self.query_multihead_attn(query_data.unsqueeze(0), None)
        att_query_data = query_data + attn_out.squeeze(0)
        query_feat = torch.max(att_query_data, dim=0).values.unsqueeze(0)

        # Tile query over refrence and merge

        query_feat = query_feat.tile((ref_data.size(0), 1))

        ref_query = torch.cat((ref_data, query_feat), dim=1)
        ref_query = self.merge(ref_query)

        attn_ref_query = self.query_ref_multihead_attn(ref_query.unsqueeze(0), None)

        feat = torch.max(ref_query + attn_ref_query.squeeze(0), dim=0).values
        loc = self.regress(feat.unsqueeze(0))
        return None, None, loc


class GATATTMLPModule(torch.nn.Module):
    def __init__(self, num_heads, n_classes=42, n_layers=2, out_size=64, split=True, only_query=False, edge_dim_in=2):
        super(GATATTMLPModule, self).__init__()
        self.only_query = only_query
        self.n_layers = n_layers
        self.split = split
        node_in_size = n_classes + 2
        self.softmax = torch.nn.Softmax(dim=-1)
        self.edge_lin = torch.nn.Linear(edge_dim_in, out_size)
        self.node_proj = torch.nn.Linear(node_in_size, out_size)
        # We use two GAT layers
        self.conv1 = EGATConv(in_node_feats=out_size, out_node_feats=2 * out_size, num_heads=num_heads,
                              in_edge_feats=out_size, out_edge_feats=2 * out_size)
        self.conv2 = EGATConv(in_node_feats=2 * out_size * num_heads, out_node_feats=4 * out_size, num_heads=num_heads,
                              in_edge_feats=2 * out_size * num_heads, out_edge_feats=out_size)
        if self.split:
            self.ref_query_conv = EGATConv(in_node_feats=out_size, out_node_feats=2 * out_size, num_heads=num_heads,
                                           in_edge_feats=out_size, out_edge_feats=out_size * 2)

        # To run multiple times the network we need to fuse all the heads outputs into one
        self.lin1 = torch.nn.Linear(num_heads * out_size, out_size)  # To fuse the edges
        self.lin2 = torch.nn.Linear(num_heads * 4 * out_size, out_size)  # To fuse the nodes
        self.lin2a = torch.nn.Linear(num_heads * 2 * out_size, out_size)  # To fuse the nodes
        self.lin3 = torch.nn.Linear(num_heads * 2 * out_size, out_size)  # To fuse the nodes
        self.line_predict = torch.nn.Linear(out_size, 1)

        self.regress = torch.nn.Sequential(
            torch.nn.Linear(out_size, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, out_size),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size, 3)
        )
        self.query_multihead_attn = MultiheadAttention(out_size, int(out_size), 4)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, g_all, all_node_data, all_edge_data, n_r):

        g, ref_and_query = g_all
        node_data0 = all_node_data[0:g.num_nodes()]
        edge_data = all_edge_data[0:g.num_edges()]

        node_data = self.node_proj(node_data0)
        edge_data = self.edge_lin(edge_data)
        edge_data_link = self.edge_lin(all_edge_data[g.num_edges():])

        for step in range(self.n_layers):
            h, e = self.conv1(g, node_data, edge_data)
            h = F.relu(h)
            e = F.relu(e)
            h, e = self.conv2(g, h.view(len(node_data), -1), e.view(len(edge_data), -1))
            edge_data = F.relu(self.lin1(e.view(len(edge_data), -1)))
            node_data = F.relu(self.lin2(h.view(len(node_data), -1)))

            # Perform the Query to Reference update
            h, e3 = self.ref_query_conv(ref_and_query, node_data, edge_data_link)
            edge_data_link = F.relu(self.lin3(e3.view(len(edge_data_link), -1)))
            node_data = node_data + F.relu(self.lin2a(h.view(len(node_data), -1)))

        # edge_data = self.line_predict(edge_data)
        if self.only_query:
            node_data = node_data[n_r:]

        node_data = self.query_multihead_attn(node_data.unsqueeze(0), None).squeeze(0)
        feat = torch.max(node_data, dim=0).values
        loc = self.regress(feat.unsqueeze(0))
        return edge_data, node_data, loc

#########################################################################################################
#  Baseline Models for the Coarse Localization Task                                                     #
#########################################################################################################


class WeightedGraphConv(GraphConv):
    """
    Description
    -----------
    GraphConv with edge weights on homogeneous graphs.
    If edge weights are not given, directly call GraphConv instead.
    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    n_feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`
    """

    def forward(self, graph: DGLGraph, n_feat, e_feat=None):
        if e_feat is None:
            return super(WeightedGraphConv, self).forward(graph, n_feat)

        with graph.local_scope():
            if self.weight is not None:
                n_feat = torch.matmul(n_feat, self.weight)
            src_norm = torch.pow(
                graph.out_degrees().float().clamp(min=1), -0.5)
            src_norm = src_norm.view(-1, 1)
            dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            dst_norm = dst_norm.view(-1, 1)
            n_feat = n_feat * src_norm
            graph.ndata["h"] = n_feat
            graph.edata["e"] = e_feat
            graph.update_all(fn.src_mul_edge("h", "e", "m"),
                             fn.sum("m", "h"))
            n_feat = graph.ndata.pop("h")
            n_feat = n_feat * dst_norm
            if self.bias is not None:
                n_feat = n_feat + self.bias
            if self._activation is not None:
                n_feat = self._activation(n_feat)
            return n_feat


class GCN(torch.nn.Module):
    """
    The model we use for now is a simple module composed of two weighted GCCs. Between them, we re-inject the initial
    class one-hot encoding in the embeddings to preserve the information
    """

    def __init__(self, in_feats, hidden_size, out_feat):
        super(GCN, self).__init__()
        self.conv1 = WeightedGraphConv(in_feats, hidden_size)
        self.conv2 = WeightedGraphConv(hidden_size + in_feats, out_feat)

    def forward(self, g, inputs, edge_weights):
        h = self.conv1(g, inputs, edge_weights)
        h = torch.relu(h)
        reminder = torch.cat([h, inputs], dim=-1)
        h = self.conv2(g, reminder, edge_weights)
        h = torch.relu(h)
        h = normalize(h, p=2.0, dim=-1)
        # return torch.cat([h, inputs], dim=-1)
        return h


def select_output_nodes(output_ref_nodes_emb, full_query_emb, n_output_nodes=10):
    """ Sorts the nodes nodes of the Reference according to embedding distance to the Query embedding.

        Returns the "n_output_nodes" that have the lowest distance.
        Also returns the distance from each Ref. Graph node to the Query embedding.
    """

    # These are the differences in encoding between the Query embedding and each node on the Reference.
    # We want to select the minima of this "function".
    node_emb_distances = torch.linalg.norm(output_ref_nodes_emb - full_query_emb, dim=1)
    ref_nodes_sorted_by_similarity_to_query = torch.argsort(node_emb_distances)

    return ref_nodes_sorted_by_similarity_to_query[:n_output_nodes], node_emb_distances


def forward(model_coarse, r_graph, q_graph, n_output_nodes):
    """ Apply the GCN on both the Reference and the Query Graph, return the 'n_output_nodes' from the Reference that best match the query.
    """
    # Apply the Model_Coarse net to compute processed embeddings for each node of the Reference Graph and of Query Graph.
    output_ref_nodes_emb   = model_coarse(r_graph, r_graph.ndata['h'].type(torch.float32), r_graph.edata['x'].type(torch.float32))  # Compute the output node embeddings for Reference graph.
    output_query_nodes_emb = model_coarse(q_graph, q_graph.ndata['h'].type(torch.float32), q_graph.edata['x'].type(torch.float32))  # Compute the output node embeddings for Query graph.

    # Aggregate the node embeddings on the Query Graph into a single embedding (by averaging).
    full_query_emb = torch.mean(output_query_nodes_emb, 0)


    ################################################################
    # Compute output: a set of nodes. Each of them should be close #
    # to the centre of one of the instances of the Query Graph.    #
    ################################################################
    output_nodes, node_emb_distances = select_output_nodes(output_ref_nodes_emb, full_query_emb, n_output_nodes=n_output_nodes)

    return output_nodes, node_emb_distances, full_query_emb, output_ref_nodes_emb