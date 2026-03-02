import torch, dgl
import torch.nn as nn
from dgl.nn.pytorch import GATv2Conv, AvgPooling, SumPooling, MaxPooling
from models.extractor.ib import InterBranch


class GatInterBranch(nn.Module):
    def __init__(self, x_size, h_size, num_classes, num_layers=2, fc=True, bn=False, pool="sumpool", path="add", direction="out", fuse="weighted"):
        super(GatInterBranch, self).__init__()
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

            self.mlp2 = nn.Sequential(
                nn.Linear(2 * h_size, 2 * h_size),
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

            self.mlp2 = nn.Sequential(
                nn.Linear(2 * h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )

        # 两层 GATv2 试一下
        if bn:
            for i in range(num_layers):
                conv = GATv2Conv(self.h_size, self.h_size, 12)
                bn = nn.BatchNorm1d(h_size)
                self.convs.append(conv)
                self.batchnorm.append(bn)
        else:
            for i in range(num_layers):
                conv = GATv2Conv(self.h_size, self.h_size, 12)
                self.convs.append(conv)

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

        # self.layernorm = nn.LayerNorm()
        # self.cat_linear = nn.Linear(h_size * 2, h_size)
        # inter-branch attention
        self.inter_branch = InterBranch(embed_dim=self.h_size, num_heads=8, batch_first=True, path=path, direction=direction)
        self.r = nn.Parameter(torch.tensor(0.5))
        self.fuse = fuse


    def forward_backbone(self, batch):
        # a batch of graphs is put in to gpu for computation
        g = batch.graph.to(torch.device("cuda:{}".format(0)))
        # to heterogenous graph (异源图)
        # 只留下edges()目的：只需要边的连接结构和节点嵌入（feats），而不需要原始图的节点或边属性这一步可以确保只保留连接信息，从而避免复杂的图结构导致的错误或不匹配
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding

        # avoid zero in-degree nodes
        g = dgl.add_self_loop(g)

        feats = self.mlp1(batch.feats.cuda())
        feats_copy = feats.clone()
        # print(f"At this step, batch size is {feats.shape}")
        for i in range(self.num_layers):
            # print(f"In loop, feats size is {feats.shape}")

            feats = self.convs[i](g, feats)
            # print(f"After Gatconv, feats size is {feats.shape}")
            feats = feats.max(dim=1)[0]

            feats = self.batchnorm[i](feats)
            # print(f"in fwd, feats size is {feats.shape}")
        # 在这一步上使用 SFA
        # feats = self.calculate(feats)
        # print(f"At this step, batch size is {feats.shape}")
        # shape eror
        temp_logits = self.pool(batch.graph.to(torch.device("cuda")), feats)
        # 返回B个图进行lstm操作后的 [B, D]  like readout function

        g.ndata["feats"] = feats_copy

        # propagate
        g = dgl.remove_self_loop(g)
        out = self.inter_branch(g, batch.max_leaf_len, batch.offset_leaf)

        # out = g.ndata.pop("feats")

        return temp_logits, out

    def forward(self, batch):
        logits, out = self.forward_backbone(batch)
        # cat_tensor = self.mlp2(torch.cat((logits, out), dim=1))
        # print(f"after forward logits size is {logits.shape}")
        # normalized_logits = F.normalize(logits, p=2, dim=0)  # p=2 表示L2范数归一化
        # normalized_out = F.normalize(out, p=2, dim=0)
        # print(logits)
        if self.fc:
            pass
            return logits
        else:
            # return logits
            if self.fuse == "weighted":
                return logits * self.r + out * (1-self.r)
            else:
                return torch.add(logits, out)