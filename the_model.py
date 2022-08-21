import sys
import torch
from abc import ABC
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun


class TopoSpaceGraphConvolutionLayer(nn.Module, ABC):
    def __init__(self, adj: torch.Tensor, in_dim: int, embed_dim: int, act=fun.relu, bias=False):
        super(TopoSpaceGraphConvolutionLayer, self).__init__()
        self.laplace = TopoSpaceGraphConvolutionLayer.diffuse_laplace(adj=adj)
        self.act = act
        self.lm = nn.Linear(in_dim, embed_dim, bias=bias)

    @staticmethod
    def diffuse_laplace(adj: torch.Tensor):
        d_x = torch.diag(torch.pow(torch.add(torch.sum(adj, dim=1), 1), -0.5))
        d_y = torch.diag(torch.pow(torch.add(torch.sum(adj, dim=0), 1), -0.5))
        adj = torch.mm(torch.mm(d_x, adj), d_y)
        return adj

    def forward(self, opposite_x: torch.Tensor):
        collected_opposite_x = self.act(self.lm(torch.mm(self.laplace, opposite_x)))
        return collected_opposite_x

class SelfFeature(nn.Module, ABC):
    def __init__(self, adj: torch.Tensor, in_dim: int, embed_dim: int, bias=False, act=fun.relu):
        super(SelfFeature, self).__init__()
        self.laplace = SelfFeature.self_laplace(adj=adj)
        self.lm = nn.Linear(in_dim, embed_dim, bias=bias)
        self.act = act

    @staticmethod
    def self_laplace(adj: torch.Tensor):
        d = torch.pow(torch.add(torch.sum(adj, dim=1), 1), -1)
        d = torch.diag(torch.add(d, 1))
        # print(d)
        return d

    def forward(self, x: torch.Tensor):
        x = self.act(self.lm(torch.mm(self.laplace, x)))
        # print("self",x)
        return x


class LinearCorrDecoder(nn.Module, ABC):
    def __init__(self, embed_dim: int, kernel_dim: int, alpha: float):
        super(LinearCorrDecoder, self).__init__()
        self.lm_x = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.lm_y = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.alpha = alpha

    @staticmethod
    def corr_x_y(x: torch.Tensor, y: torch.Tensor):
        assert x.size()[1] == y.size()[1], "Different size!"
        x = torch.sub(x, torch.mean(x, dim=1).view([-1, 1]))
        y = torch.sub(y, torch.mean(y, dim=1).view([-1, 1]))
        lxy = torch.mm(x, y.t())
        lxx = torch.diag(torch.mm(x, x.t()))
        lyy = torch.diag(torch.mm(y, y.t()))
        std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
        corr = torch.div(lxy, std_x_y+0.000001)
        corr[corr > 1] = 1
        corr[corr < -1] = -1
        return corr

    @staticmethod
    def scale_sigmoid_activation_function(x: torch.Tensor, alpha: int or float):
        torch.all(x.ge(-1)) and torch.all(x.le(1)), "Out of range!"
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        x = torch.sigmoid(torch.mul(alpha, x))
        max_value = torch.sigmoid(alpha)
        min_value = torch.sigmoid(-alpha)
        output = torch.div(torch.sub(x, min_value), torch.sub(max_value, min_value))
        return output

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.lm_x(x)
        y = self.lm_y(y)
        out = LinearCorrDecoder.corr_x_y(x=x, y=y)
        out = LinearCorrDecoder.scale_sigmoid_activation_function(x=out, alpha=self.alpha)
        return out


class GraphConvolution(nn.Module, ABC):
    def __init__(self, adj: torch.Tensor,  x_feature: torch.Tensor, y_feature: torch.Tensor,  mask: torch.Tensor,
                 embed_dim: int, kernel_dim: int, **kwargs):
        super(GraphConvolution, self).__init__()
        self.adj = adj
        self.x = x_feature
        self.y = y_feature
        self.mask = mask
        self.embed_dim = embed_dim
        self.kernel_dim = kernel_dim
        self.act = kwargs.get("act", fun.relu)
        self.alpha = kwargs.get("alpha", 5.74)
        self.beta = kwargs.get("beta", 1.75)
        self.self_space = self._add_self_space()
        self.topo_space = self._add_topo_space()
        self.predict_layer = self._add_predict_layer()

    def _add_self_space(self):
        x = SelfFeature(adj=self.adj, in_dim=self.x.size()[1], embed_dim=self.embed_dim)
        y = SelfFeature(adj=self.adj.t(), in_dim=self.y.size()[1], embed_dim=self.embed_dim)
        return nn.ModuleList([x, y])

    def _add_topo_space(self):
        x = TopoSpaceGraphConvolutionLayer(adj=self.adj, in_dim=self.y.size()[1], embed_dim=self.embed_dim)
        y = TopoSpaceGraphConvolutionLayer(adj=self.adj.t(), in_dim=self.x.size()[1], embed_dim=self.embed_dim)
        return nn.ModuleList([x, y])

    def _add_predict_layer(self):
        return LinearCorrDecoder(embed_dim=self.embed_dim, kernel_dim=self.kernel_dim, alpha=self.alpha)
        # return BilinearDecoder(embed_dim=self.embed_dim, kernel_dim=self.kernel_dim)

    def loss_fun(self, predict: torch.Tensor):
        true_data = torch.masked_select(self.adj, self.mask)
        predict = torch.masked_select(predict, self.mask)
        beta_weight = torch.empty_like(true_data).fill_(self.beta)
        back_betaweight = torch.ones_like(true_data)
        weight = torch.where(true_data.eq(1), beta_weight, back_betaweight)
        bce_loss = nn.BCELoss(weight=weight, reduction="mean")
        return bce_loss(predict, true_data)

    def forward(self):
        self_layer_x, self_layer_y = [layer for layer in self.self_space]
        self_x = self_layer_x(x=self.x)
        self_y = self_layer_y(x=self.y)

        topo_layer_x, topo_layer_y = [layer for layer in self.topo_space]
        collect_opposite_x = topo_layer_x(opposite_x=self.y)
        collect_opposite_y = topo_layer_y(opposite_x=self.x)

        x = torch.stack([self_x, collect_opposite_x], dim=0)
        y = torch.stack([self_y, collect_opposite_y], dim=0)

        x = torch.sum(x, dim=0)
        y = torch.sum(y, dim=0)
        return self.predict_layer(x=x, y=y)







