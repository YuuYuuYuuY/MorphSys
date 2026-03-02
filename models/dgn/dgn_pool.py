import torch, dgl
import torch.nn as nn
from dgl.nn.pytorch import DGNConv, SumPooling, AvgPooling, MaxPooling
from torch.nn.functional import relu
from dgl import LaplacianPE
from utils import get_root_logger


class DgnPool(nn.Module):
    def __init__(self, x_size, h_size, num_classes, num_layers=2, fc=True, bn=False, pool="sumpool", path="add", direction="out", fuse="weighted"):
        super(DgnPool, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        self.num_layers = num_layers
        # gnn layers
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()

        # mlp 映射到128维空间
        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.BatchNorm1d(2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )

        if bn:
            for i in range(num_layers):
                conv = DGNConv(in_size=h_size, out_size=h_size,  aggregators=['dir1-av', 'dir1-dx', 'sum'], scalers=['identity', 'amplification'], delta=2.5)
                bn = nn.BatchNorm1d(h_size)
                self.convs.append(conv)
                self.batchnorm.append(bn)
        else:
            for i in range(num_layers):
                conv = DGNConv(in_size=h_size, out_size=h_size,  aggregators=['dir1-av', 'dir1-dx', 'sum'], scalers=['identity', 'amplification'], delta=2.5)
                self.convs.append(conv)
        # pooling layer as readout function
        # pooling layer as readout function
        if pool == "sumpool":
            self.pool = SumPooling()
        elif pool == "avgpool":
            self.pool = AvgPooling()
        elif pool == "maxpool":
            self.pool = MaxPooling()

        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward_backbone(self, batch):
        # a batch of graphs is put in to gpu for computation
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph (异源图)
        transform = LaplacianPE(k=3, feat_name='eig')
        g = transform(g)

        eig = g.ndata['eig']

        g = dgl.graph(g.edges())
        n = g.number_of_nodes()



        # feed embedding
        feats = self.mlp1(batch.feats.cuda())
        # print(f"feats shape: {feats.shape}")
        # print(f"At this step, batch size is {feats.shape}")
        for i in range(self.num_layers):
            # print(f"At this step, feats size is {feats.shape}")
            feats = self.convs[i](g, feats, eig_vec=eig)
            # print(f"After ginconv, feats size is {feats.shape}")
            feats = self.batchnorm[i](feats)
            # print(f"in fwd, feats size is {feats.shape}")
        # 在这一步上使用 SFA
        # feats = self.calculate(feats)
        # print(f"At this step, batch size is {feats.shape}")
        # shape eror
        logits = self.pool(batch.graph.to(torch.device("cuda")), feats)
        # 返回B个图进行lstm操作后的 [B, D]  like readout function
        return logits

    def forward(self, batch):
        logits = self.forward_backbone(batch)
        # print(f"after forward logits size is {logits.shape}")
        # print(logits)
        if self.fc:
            pass
            return logits
        else:
            return logits
