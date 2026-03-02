import torch, dgl
import torch.nn as nn
from dgl.nn.pytorch import GINConv, SumPooling, AvgPooling, MaxPooling
from torch.nn.functional import relu
from utils import get_root_logger



class GinPool(nn.Module):
    def __init__(self, x_size, h_size, num_classes, num_layers=2, fc=True, bn=False, pool="sumpool", path="add", direction="out", fuse="weighted"):
        super(GinPool, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        self.num_layers = num_layers
        # gnn layers
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        if bn:
            for i in range(num_layers):
                if i:
                    mlp = nn.Sequential(nn.Linear(self.h_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                else:
                    mlp = nn.Sequential(nn.Linear(self.x_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                conv = GINConv(mlp, learn_eps=True, activation=relu)
                bn = nn.BatchNorm1d(h_size)
                self.convs.append(conv)
                self.batchnorm.append(bn)
        else:
            for i in range(num_layers):
                if i:
                    mlp = nn.Sequential(nn.Linear(self.h_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                else:
                    mlp = nn.Sequential(nn.Linear(self.x_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                conv = GINConv(mlp, learn_eps=True)
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
        # 只留下edges()目的：只需要边的连接结构和节点嵌入（feats），而不需要原始图的节点或边属性这一步可以确保只保留连接信息，从而避免复杂的图结构导致的错误或不匹配
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        feats = batch.feats.cuda()
        # print(f"At this step, batch size is {feats.shape}")
        for i in range(self.num_layers):
            # print(f"At this step, feats size is {feats.shape}")
            feats = self.convs[i](g, feats)
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


