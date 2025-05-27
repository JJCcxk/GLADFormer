import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    ## type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class Contrassive_Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 用于 global 引导
        self.fusion = nn.Sequential(
            nn.Conv2d(n_fea_middle + n_fea_middle, n_fea_middle, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img: [B, 3, H, W]
        mean_c = img.mean(dim=1, keepdim=True)     # [B, 1, H, W]
        input = torch.cat([img, mean_c], dim=1)    # [B, 4, H, W]

        x_1 = self.conv1(input)                    # [B, C, H, W]
        local_feat = self.depth_conv(x_1)          # 局部空间光照特征

        global_feat = self.global_pool(local_feat).expand_as(local_feat)  # 复制成局部结构
        illu_fea = self.fusion(torch.cat([local_feat, global_feat], dim=1))  # 融合局部+全局光照响应

        illu_map = self.conv2(illu_fea)               
        return illu_fea, illu_map                     


class LCAM(nn.Module):
    def __init__(self, dim, dim_head,heads, window_size=8, chunk_size=64):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.scale = dim_head ** -0.5

        # 全局Token生成器
        self.global_token = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # -> [B, C, 1, 1]
            nn.Flatten(start_dim=1),          # -> [B, C]
            nn.Linear(dim, dim)               # -> [B, C]
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        )

    def forward(self, x, illu_fea_trans=None):
        B, H, W, C = x.shape
        Ws = self.window_size
        pad_h = (Ws - H % Ws) % Ws
        pad_w = (Ws - W % Ws) % Ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode="reflect")
        if illu_fea_trans is not None:
            illu_fea_trans = F.pad(illu_fea_trans, (0, 0, 0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[1], x.shape[2]
        num_windows = (Hp // Ws) * (Wp // Ws)

        # window partition
        x_windows = x.view(B, Hp // Ws, Ws, Wp // Ws, Ws, C)
        x_windows = x_windows.permute(0,1,3,2,4,5).reshape(B * num_windows, Ws * Ws, C)

        if illu_fea_trans is not None:
            illu_windows = illu_fea_trans.view(B, Hp // Ws, Ws, Wp // Ws, Ws, C)
            illu_windows = illu_windows.permute(0,1,3,2,4,5).reshape(B * num_windows, Ws * Ws, C)

        # QKV projection
        qkv = self.qkv(x_windows)
        T = x_windows.size(1)
        qkv = qkv.reshape(B * num_windows, T, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if illu_fea_trans is not None:
            illu = illu_windows.reshape(B * num_windows, T, self.num_heads, self.dim_head).permute(0,2,1,3)
            v = v * (1 + torch.sigmoid(illu))

        # 全局 token 生成并扩展到每个窗口
        global_token = self.global_token(x.permute(0, 3, 1, 2)).view(B, self.num_heads, 1, self.dim_head)
        global_token_k = global_token.expand(-1, -1, num_windows, -1).reshape(B * num_windows, self.num_heads, 1, self.dim_head)
        global_token_v = global_token_k.clone()
        k = torch.cat([k, global_token_k], dim=2)
        v = torch.cat([v, global_token_v], dim=2)
        
        # chunked attention
        Bn, heads, T, d = q.shape
        outs = []
        for i in range(0, T, self.chunk_size):
            q_chunk = q[:, :, i:i+self.chunk_size, :]
            attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            out_chunk = torch.matmul(attn_chunk, v)
            outs.append(out_chunk)
        out = torch.cat(outs, dim=2)

        out = out.transpose(1, 2).reshape(B * num_windows, T, C)
        out = self.proj(out)
        out = out.view(B, Hp // Ws, Wp // Ws, Ws, Ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        out = out[:, :H, :W, :]

        # 深度卷积位置编码
        pos = self.pos_emb(x.view(B, Hp, Wp, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pos = pos[:, :H, :W, :]

        return out + pos



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class LGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LCAM(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                LGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = LGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                LGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution,         LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class PAGM(nn.Module):
    def __init__(self, codebook_dim=64, num_embeddings=128, in_channels=3):
        super().__init__()
        # 可训练 codebook：形状 [K, D]
        self.codebook = nn.Embedding(num_embeddings, codebook_dim)

        # 预测每个像素属于 K 个向量的 soft 权重（Soft assignment）
        self.code_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 30, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(30, num_embeddings, 1)  # 输出每像素的 K 维权重
        )

        # 通道注意力（简化 SE block）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

        # 融合 gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(codebook_dim + in_channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # 残差 refinement
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(codebook_dim + in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, base_map):
        """
        x: 输入特征，增强前图像，形状 [B, C, H, W]
        base_map: 用于 codebook soft query，形状 [B, C, H, W]
        """
        B, C, H, W = base_map.shape
        attn_weights = self.channel_attn(base_map)
        weighted_x = x * attn_weights

        # soft assignment to codebook
        code_logits = self.code_predictor(base_map)            # [B, K, H, W]
        code_weights = F.softmax(code_logits, dim=1)           # soft query weights
        codebook_vectors = self.codebook.weight                # [K, D]

        # weighted sum over codebook vectors: einsum over (bkxy, kd) -> bdxy
        code_feat = torch.einsum('bkhw,kd->bdhw', code_weights, codebook_vectors)

        # 拼接融合
        fused = torch.cat([weighted_x, code_feat], dim=1)      # [B, C+D, H, W]
        gate = self.fusion_gate(fused)
        fused = fused * gate

        delta = self.refine_blocks(fused)
        return x + torch.tanh(delta) * self.res_scale
    
class GLADFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(GLADFormer_Single_Stage, self).__init__()
        self.estimator = Contrassive_Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
        self.pixel_refiner = PAGM(
            codebook_dim=128,
            num_embeddings=256
        )
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)
        output_img = self.pixel_refiner(output_img,illu_map)

        return output_img


class GLADFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=3, num_blocks=[1,1,1]):
        super(GLADFormer, self).__init__()
        self.stage = stage

        modules_body = [GLADFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out




if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = GLADFormer(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
    #print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

