import copy
import torch
from torch_geometric.nn.models import MLP
from torch_geometric.nn import global_mean_pool, TransformerConv, BatchNorm, TopKPooling


class UniMP(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            num_layers: int,
            num_heads: int,
            metadata: tuple,
            **kwargs
    ):
        super().__init__()
        self.node_lins = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_lins[node_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.edge_lins = torch.nn.ModuleDict()
        for edge_type in set([et[1] for et in metadata[1]]):
            edge_type = '$%s$' % edge_type
            self.edge_lins[edge_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // num_heads,
                edge_dim=hidden_channels,
                heads=num_heads,
                beta=True,
            ))
            self.bns.append(BatchNorm(in_channels=hidden_channels))

        self.pool_in_lin = MLP(
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            norm=None,
        )
        self.pool = TopKPooling(
            in_channels=hidden_channels,
            min_score=1e-3,
        )

        self.out_lin = MLP(
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            norm=None,
        )

    def forward(self, data0, **kwargs):
        data = copy.deepcopy(data0)
        node_types, edge_types = data.metadata()
        for node_type in node_types:
            x = data[node_type].x
            data[node_type].x = self.node_lins[node_type](x)

        # aligning the edge feature length
        for edge_type in edge_types:
            edge_attr = data[edge_type].edge_attr
            edge_attr = self.edge_lins['$%s$' % edge_type[1]](edge_attr)
            data[edge_type].edge_attr = edge_attr

        # conv operators
        data = data.to_homogeneous()
        for i, conv in enumerate(self.convs):
            data.x = conv(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
            )
            data.x = self.bns[i](data.x)

        # return the result
        data.x = self.pool_in_lin(data.x)
        data.x, data.edge_index, _, data.batch, perm, score = self.pool(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        emb = global_mean_pool(data.x, data.batch)
        emb = self.out_lin(emb)
        if kwargs.get('score') and kwargs['score'] is True:
            score, indices = torch.sort(score, descending=True)
            return emb, data.src[perm[indices]], score
        return emb
