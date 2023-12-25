import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

NUM_CLASS = 16

#main
class models(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.DT = DT(in_channels, num_tokens, dim, depth, heads, dropout, emb_dropout) # Deep Feature Extraction part
        self.Resdd = D(num_classes, dim, depth)  # Controllable Mapping part

    def forward(self, x):
        x = self.DT(x)
        x = self.Resdd(x)
        return x

# Deep Feature Extraction
class DT(nn.Module):
    def __init__(self, in_channels, num_tokens, dim, depth, heads, dropout, emb_dropout):
        super(DT, self).__init__()
        self.L = num_tokens
        self.cT = dim

        self.Convolution = Convolution(in_channels)  # Convolution module
        self.Tokenizer = Tokenizer(num_tokens, dim, emb_dropout)  # Tokenizer module
        self.SelfAttention = SelfAttention(dim, depth, heads,  dropout)  # Attention module

        self.to_cls_token = nn.Identity()


    def forward(self, x, mask=None):
        x = self.Convolution(x)
        x = self.Tokenizer(x)
        x = self.SelfAttention(x, mask)  
        x = self.to_cls_token(x[:, 0])

        return x

# Controllable Mapping 
class D(nn.Module):
    def __init__(self, num_classes, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([
                Residual(LayerNormalize(dim, ddNet()))
                for _ in range(depth)
            ])
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x):
        for dd in self.layers:
            x = dd(x)  # go to ddNet
            x = self.nn1(x)
        return x

# Convolution module
class Convolution(nn.Module):
    def __init__(self, in_channels):
        super(Convolution, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=16*28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
    def forward(self, x):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')

        return x

# Tokenizer module
class Tokenizer(nn.Module):
    def __init__(self, num_tokens, dim, emb_dropout):
        super(Tokenizer, self).__init__()
        self.L = num_tokens
        self.cT = dim
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        return x

# Attention module
class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)

        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softma
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out
class SelfAttention(nn.Module):
    def __init__(self, dim, depth, heads,  dropout):
        super().__init__()
        self.layers = (nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout)))
                for _ in range(depth)
            ]))

    def forward(self, x, mask=None):
        for attention in self.layers:
            x = attention(x, mask=mask)  # go to attention
        return x

# DD
class ddNet(nn.Module):

    def __init__(self):
        super(ddNet, self).__init__()

        # xw+b
        self.fc0 = nn.Linear(64, 8, bias=False)
        self.dd = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 64, bias=False)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        x = self.fc0(x)
        c=x

        for i in range(2):
            x=self.dd(x)*c

        x = self.fc2(x)
        return x

# Residual connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# Layer Normalization
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


if __name__ == '__main__':
    model = models()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())

