import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

## 1x1 convolution layer가 pixel-wise linear layer라고 볼 수 있당.
class Linear_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear_embedding, self).__init__()
        self.linear_2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.linear_2d(x)

class LayerNorm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dims))
        self.b_2 = nn.Parameter(torch.zeros(dims))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLP(nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels))
        
    def forward(self, x):
        return self.mlp(x)

class Patch_merging(nn.Module):
    def __init__(self, channels, out_channels):
        super(Patch_merging, self).__init__()
        self.linear = nn.Linear(channels, out_channels)
        
    def forward(self, feature, height, scale=2):
        feature = rearrange(feature, 'b (h p1 w p2) c -> b (h w) (p1 p2 c)', h = height, p1 = scale, p2 = scale)
        return self.linear(feature)

class Window_MSA(nn.Module):
    def __init__(self, channels, window_size = 8, is_shift=False, n_head=8):
        super(Window_MSA, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(channels, channels) for _ in range(4)])
        self.window_size = window_size
        self.is_shift = is_shift
        self.n_head = n_head
        
        # relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, n_head))
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords, coords]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords += window_size - 1  # shift to start from 0
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x, height):
        # x : b, h * w, c
        _, hw, _ = x.shape
        stride = self.window_size // 2
        mask = torch.zeros((height, hw // height), device=x.device, dtype=torch.uint8)
        if self.is_shift:
            mask[:-stride, -stride:] = 1
            mask[-stride:, :-stride] = 2
            mask[-stride:, -stride:] = 3
            x = rearrange(x, 'b (h w) c -> b h w c', h=height)
            temp = torch.clone(x)
            temp[:, :-stride, :-stride, :] = x[:, stride:, stride:, :]
            temp[:, -stride:, :-stride, :] = x[:, :stride, stride:, :]
            temp[:, :-stride, -stride:, :] = x[:, stride:, :stride, :]
            temp[:, -stride:, -stride:, :] = x[:, :stride, :stride, :]
            temp = rearrange(temp, 'b h w c -> b (h w) c')
            x = temp
        mask = mask.view(1, -1)
        
        query, key, value = [l(x) for l in self.linears[:-1]]
        
        query = rearrange(query, 'b (h p1 w p2) (n_h c) -> b (h w) n_h (p1 p2) c', h=height//self.window_size, p1=self.window_size, p2=self.window_size, n_h = self.n_head)
        key = rearrange(key, 'b (h p1 w p2) (n_h c) -> b (h w) n_h (p1 p2) c', h=height//self.window_size, p1=self.window_size, p2=self.window_size, n_h = self.n_head)
        value = rearrange(value, 'b (h p1 w p2) (n_h c) -> b (h w) n_h (p1 p2) c', h=height//self.window_size, p1=self.window_size, p2=self.window_size, n_h = self.n_head)
        mask = rearrange(mask, 'b (h p1 w p2) -> b (h w) (p1 p2)', h=height//self.window_size, p1=self.window_size, p2=self.window_size)
        mask = mask[:, :, :, None] - mask[:, :, None, :]
        mask = mask == 0
        mask = torch.unsqueeze(mask, dim=2)
        
        result = self.attention(query, key, value, mask)
        result = rearrange(result, 'b (h w) n_h (p1 p2) c -> b (h p1 w p2) (n_h c)', h=height//self.window_size, p1=self.window_size, p2=self.window_size, n_h = self.n_head)
        
        return result
        
    def attention(self, query, key, value, mask):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        c = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(c) + relative_position_bias.unsqueeze(0).unsqueeze(0)
        scores *= mask
        p_attn = F.softmax(scores, dim=-1)
        result = torch.matmul(p_attn, value)
        return result

class Swin_transformer_block(nn.Module):
    def __init__(self, channels, window_size):
        super(Swin_transformer_block, self).__init__()
        self.patch_merging = Patch_merging(2*channels, channels)
        self.layer_norms = nn.ModuleList([LayerNorm(channels) for _ in range(4)])
        self.mlp_layers = nn.ModuleList([MLP(channels) for _ in range(2)])
        self.W_MSA = Window_MSA(channels, window_size)
        self.SW_MSA = Window_MSA(channels, window_size, is_shift=True)
        
    def forward(self, x, height):
        # x : b, h * w, C
        
        # block 1
        x = self.W_MSA(self.layer_norms[0](x), height) + x
        x = self.mlp_layers[0](self.layer_norms[1](x)) + x
        
        # block 2
        x = self.SW_MSA(self.layer_norms[2](x), height) + x
        x = self.mlp_layers[1](self.layer_norms[3](x)) + x
        
        return x

class Swin_transformer(nn.Module):
    def __init__(self, patch_size = 4, window_size = 8, merge_size = 2, model_dim = 128, num_layers_in_stage = [2, 2, 6, 2]):
        '''
            patch_size : 입력으로 받은 이미지를 패치단위로 합칠 때, 패치 사이즈 ===> 반드시 이미지 사이즈에 나누어 떨어져야 한다.
            window_size : 하나의 윈도우의 사이즈 ===> 2의 배수여야 하고, 가장 작은 feature map 크기보다 크거나 동일해야 한다.
            merge_size : patch merge 하는 이미지 사이즈, 이미지 축소 사이즈
            model_dim : 입력으로 받은 이미지를 패치 단위로 합치고 Tranformer model에 들어가는 데이터의 차원
            num_layers_in_stages : stage 마다 Swin Transformer blocks의 숫자
        '''
        super(Swin_transformer, self).__init__()
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.linear_embedding = Linear_embedding(self.patch_size ** 2 * 3, model_dim)
        
        self.swin_transformer_blocks_per_stage = nn.ModuleList([])
        self.patch_merging_per_stage = nn.ModuleList([])
        for i, num_layers in enumerate(num_layers_in_stage):
            if i > 0:
                patch_merging = Patch_merging(4 * channels, 2 * channels)
                self.patch_merging_per_stage.append(patch_merging)

            channels = model_dim * merge_size ** (i)
            swin_transformer_blocks = nn.ModuleList([Swin_transformer_block(channels, window_size) for _ in range(num_layers)])
            self.swin_transformer_blocks_per_stage.append(swin_transformer_blocks)
        
    def patch_partition(self, img, scale=2):
        return rearrange(img, 'b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = scale, p2 = scale)
    
    def forward(self, x):
        # x : b, c, h, w
        _, _, h, _ = x.shape
        patch_partitioned_x = self.patch_partition(x, scale = self.patch_size)
        h = h // self.patch_size
        features = {}
        
        # embedded_x : b, model_dim, h/patch_size, w/patch_size
        embedded_x = self.linear_embedding(patch_partitioned_x)
        
        # feature : b, h * w / patch_size ** 2, model_dim
        feature = torch.flatten(embedded_x, 2, -1).transpose(1, 2).contiguous()
        
        for swin_transformer_block in self.swin_transformer_blocks_per_stage[0]:
            feature = swin_transformer_block(feature, height=h)
        features['res2'] = rearrange(feature, 'b (h w) c -> b c h w', h=h)
        
        for i, (patch_merging, swin_transformer_blocks) in enumerate(zip(self.patch_merging_per_stage, self.swin_transformer_blocks_per_stage[1:])):
            feature = patch_merging(feature, h)
            h = h // self.merge_size
            
            for swin_transformer_block in swin_transformer_blocks:
                feature = swin_transformer_block(feature, height=h)
            features['res' + str(i+3)] = rearrange(feature, 'b (h w) c -> b c h w', h=h)
        
        # feature = rearrange(feature, 'b (h w) c -> b c h w', h = h)
        return features