import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from model import common
from model import acb
from model.diversebranchblock import DiverseBranchBlock

def make_model(args, parent=False):
    return ALAN(args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_acb=True, use_dbb=False, deploy=False, acb_norm="batch"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features, use_acb=use_acb, use_dbb=use_dbb, deploy=deploy, acb_norm=acb_norm)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class LKA(nn.Module):
    def __init__(self, dim, dw_ker=5, dwd_ker=7, dwd_pad=9, dwd_dil=3, use_acb=True, use_dbb=False, deploy=False, acb_norm="batch", use_attn=True):
        super().__init__()
        if use_acb:
            if use_dbb:
                self.conv0 = DiverseBranchBlock(dim, dim, dw_ker, padding=(dw_ker-1)//2, groups=dim, deploy=deploy)
                # self.conv_spatial = DiverseBranchBlock(dim, dim, dwd_ker, stride=1, padding=dwd_pad, groups=dim, dilation=dwd_dil, deploy=deploy)  # 不能处理 padding != kernel_size // 2的情况
                self.conv_spatial = nn.Conv2d(dim, dim, dwd_ker, stride=1, padding=dwd_pad, groups=dim, dilation=dwd_dil)
            else:
                self.conv0 = acb.ACBlock(dim, dim, dw_ker, padding=(dw_ker-1)//2, groups=dim, deploy=deploy, norm=acb_norm)
                self.conv_spatial = acb.ACBlock(dim, dim, dwd_ker, stride=1, padding=dwd_pad, groups=dim, dilation=dwd_dil, deploy=deploy, norm=acb_norm)
        else:
            self.conv0 = nn.Conv2d(dim, dim, dw_ker, padding=(dw_ker-1)//2, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, dwd_ker, stride=1, padding=dwd_pad, groups=dim, dilation=dwd_dil)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.use_multi = use_attn

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        if self.use_multi:
            return u * attn
        else:
            return attn


class Attention(nn.Module):
    def __init__(self, d_model, dw_ker=5, dwd_ker=7, dwd_pad=9, dwd_dil=3, use_acb=True, use_dbb=False, deploy=False, acb_norm="batch", use_attn=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model, dw_ker=dw_ker, dwd_ker=dwd_ker, dwd_pad=dwd_pad, dwd_dil=dwd_dil, use_acb=use_acb, use_dbb=use_dbb, deploy=deploy, acb_norm=acb_norm, use_attn=use_attn)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, dw_ker=5, dwd_ker=7, dwd_pad=9, dwd_dil=3, bb_norm="BN", use_acb=True, use_dbb=False, deploy=False, acb_norm="batch", use_attn=True):
        super().__init__()
        # if acb_norm not in ["batch", "layer", "no", "v8old", "inst"]:
        #     raise NotImplementedError 
        # if bb_norm not in ["BN", "LN", "no"]:
        #     raise NotImplementedError 
        self.bb_norm = bb_norm
        if bb_norm == "BN":
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        elif bb_norm == "LN":
            # self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            # self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            # self.norm1 = LayerNorm(dim)
            # self.norm2 = LayerNorm(dim)
            self.norm1 = LayerNorm(dim, data_format="channels_first")
            self.norm2 = LayerNorm(dim, data_format="channels_first")
        self.attn = Attention(dim, dw_ker=dw_ker, dwd_ker=dwd_ker, dwd_pad=dwd_pad, dwd_dil=dwd_dil, use_acb=use_acb, use_dbb=use_dbb, deploy=deploy, acb_norm=acb_norm, use_attn=use_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = nn.BatchNorm2d(dim)
        # self.norm2 = LayerNorm(dim, data_format="channels_first")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, use_acb=use_acb, use_dbb=use_dbb, deploy=deploy, acb_norm=acb_norm)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        skip = x
        if self.bb_norm == "BN":
            x = self.norm1(x)
        elif self.bb_norm == "LN":
            # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm1(x)
            # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = skip + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        
        skip = x
        if self.bb_norm == "BN":
            x = self.norm2(x)
        elif self.bb_norm == "LN":
            # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm2(x)
            # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = skip + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, bb_norm="BN", use_acb=True, use_dbb=False, deploy=False, acb_norm="batch"):
        super().__init__()
        # if acb_norm not in ["batch", "layer", "no", "v8old", "inst"]:
        #     raise NotImplementedError 
        # if bb_norm not in ["BN", "LN", "no"]:
        #     raise NotImplementedError 
        if use_acb:
            if use_dbb:
                self.proj = DiverseBranchBlock(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                padding=patch_size // 2, deploy=deploy)
            else:
                self.proj = acb.ACBlock(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                padding=patch_size // 2, deploy=deploy, norm=acb_norm)
        else:
            patch_size = to_2tuple(patch_size)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                  padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.bb_norm = bb_norm
        if bb_norm == "BN":
            self.norm = nn.BatchNorm2d(embed_dim)
        elif bb_norm == "LN":
            self.norm = LayerNorm(embed_dim, data_format="channels_first")
            # self.norm = LayerNorm(embed_dim)
            # self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.bb_norm == "BN":
            x = self.norm(x)
        elif self.bb_norm == "LN":
            # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768, use_acb=True, use_dbb=False, deploy=False, acb_norm="batch"):
        super(DWConv, self).__init__()
        if use_acb:
            if use_dbb:
                self.dwconv = DiverseBranchBlock(dim, dim, 3, 1, 1, groups=dim, deploy=deploy)
            else:
                self.dwconv = acb.ACBlock(dim, dim, 3, 1, 1, groups=dim, deploy=deploy, norm=acb_norm)
        else:
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ALAN(nn.Module):
    # def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
    #             mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
    #              depths=[3, 4, 6, 3], num_stages=4, flag=False):
    def __init__(self, args):
        super(ALAN, self).__init__()

        img_size=args.patch_size
        in_chans=args.n_colors
        # num_classes=1000  # change to in_chans
        embed_dims=args.dims
        mlp_ratios=args.mlp_ratios #[4, 4, 4, 4]
        drop_rate=0.
        drop_path_rate=0.
        norm_layer=nn.LayerNorm
        depths=args.depths
        num_stages=len(depths)
        # flag=False
        use_acb = args.use_acb
        use_dbb = args.use_dbb
        conv_layer = common.default_dbb if use_acb and use_dbb else common.default_acb if use_acb else common.default_conv2d
        use_inf = args.load_inf
        acb_norm = args.acb_norm
        use_attn = not args.no_attn
        bb_norm = args.bb_norm
        # lka_kernel = args.LKAkSize  # default: 21
        dwd_dil = args.DWDdil  # default: 3
        dwd_ker = args.DWDkSize  # default: 7
        # dwd_ker = lka_kernel // dwd_dil  # default: 7
        dw_ker = 2 * dwd_dil - 1  # default: 5
        dwd_pad = ((dwd_ker - 1) // 2) * dwd_dil  # default: 9
        if args.repecn_up_feat == 0:
            num_up_feat = embed_dims[-1]
        else:
            num_up_feat = args.repecn_up_feat

        # if flag == False:
        #     self.num_classes = num_classes
        self.scale = args.scale[0]
        self.depths = depths
        self.num_stages = num_stages
        self.upsampling = args.upsampling
        self.interpolation = args.interpolation
        self.stage_res = args.stage_res
        self.bb_norm = bb_norm
        self.stage_norm = not args.no_layernorm
        self.down_fea = args.down_fea
        
        # RGB mean for DIV2K
        if in_chans == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # ##################################################################################
        # Shallow Feature Extraction.
        # if use_norm:
        #     self.shallow = nn.Sequential(
        #         acb.ACBlock(in_chans, embed_dims[0], 3, 1, 1, deploy=use_inf, norm=acb_norm),
        #         LayerNorm(embed_dims[0], eps=1e-6, data_format="channels_first")
        #         ,nn.GELU()
        #     )
        # else:
        #     self.shallow = nn.Sequential(
        #         acb.ACBlock(in_chans, embed_dims[0], 3, 1, 1, deploy=use_inf, norm=acb_norm)
        #         ,nn.GELU()
        #     )
        
        # ##################################################################################
        # Deep Feature Extraction. Must use LayerNorm, or the output just use the bicubic, with PSNR fixed
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        if self.down_fea:
            self.down_fea = nn.PixelUnshuffle(2)
            self.up_fea = nn.PixelShuffle(2)

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size, # if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            # stride=4 if i == 0 else 2,
                                            stride=1,
                                            in_chans=in_chans if i == 0 else (embed_dims[i - 1] // 4) if self.down_fea else embed_dims[i - 1],
                                            embed_dim=embed_dims[i] if i == 0 else (embed_dims[i] // 4) if self.down_fea else embed_dims[i],
                                            bb_norm=bb_norm, 
                                            use_acb=use_acb, use_dbb=use_dbb, 
                                            deploy=use_inf, acb_norm=acb_norm)

            block = nn.ModuleList([Block(
                dim=(embed_dims[i] // 4) if self.down_fea else embed_dims[i], 
                mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], dw_ker=dw_ker, dwd_ker=dwd_ker, dwd_pad=dwd_pad, dwd_dil=dwd_dil, 
                bb_norm=bb_norm, use_acb=use_acb, use_dbb=use_dbb, deploy=use_inf, acb_norm=acb_norm, use_attn=use_attn)
                for j in range(depths[i])])
            if self.stage_norm:
                norm = norm_layer((embed_dims[i] // 4) if self.down_fea else embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            if self.stage_norm:
                setattr(self, f"norm{i + 1}", norm)

        # ##################################################################################
        # Upsampling
        self.preup_norm = LayerNorm(embed_dims[-1], eps=1e-6, data_format="channels_first")
        if self.upsampling != 'PixelShuffleDirect' and self.upsampling != 'Deconv':
            self.preup = nn.Sequential(
                # DiverseBranchBlock(embed_dims[-1], num_up_feat, 3, 1, 1, deploy=use_inf) if use_acb and use_dbb else acb.ACBlock(embed_dims[-1], num_up_feat, 3, 1, 1, deploy=use_inf, norm=acb_norm) if use_acb else nn.Conv2d(embed_dims[-1], num_up_feat, 3, 1, 1)
                conv_layer(embed_dims[-1], num_up_feat, 3, 1, 1, deploy=use_inf, norm=acb_norm)
                ,nn.GELU()
            )
        # Upsampling layer.
        if self.upsampling == 'Deconv':
            # Deconvolution layer.
            self.deconv = nn.ConvTranspose2d(embed_dims[-1], in_chans, (9, 9), (self.scale, self.scale),
                                             (4, 4), (self.scale - 1, self.scale - 1))
        elif self.upsampling == 'PixelShuffleDirect':
            self.pixelshuffledirect = common.UpsamplerDirect(conv_layer, self.scale, embed_dims[-1], in_chans, deploy=use_inf, norm=acb_norm)
        elif self.upsampling == 'PixelShuffle':
            if args.no_act_ps:
                self.pixelshuffle = nn.Sequential(
                    common.Upsampler(conv_layer, self.scale, num_up_feat, act=False, deploy=use_inf, norm=acb_norm), #act='gelu' for v5
                    conv_layer(num_up_feat, in_chans, 3, 1, 1, deploy=use_inf, norm=acb_norm)
                )
            else:  # default option
                self.pixelshuffle = nn.Sequential(
                    common.Upsampler(conv_layer, self.scale, num_up_feat, act='gelu', deploy=use_inf, norm=acb_norm),
                    conv_layer(num_up_feat, in_chans, 3, 1, 1, deploy=use_inf, norm=acb_norm)
                )
        elif self.upsampling == 'Nearest' or self.upsampling == 'NearestNoPA':
            # Nearest + Conv/ACBlock
            if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
                for i in range(int(math.log(self.scale, 2))):  #  循环 n 次
                    self.add_module(f'up{i}', conv_layer(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm=acb_norm))
            elif self.scale == 3:  # 缩放因子等于 3
                self.up = conv_layer(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm=acb_norm)
            else:
                # 报错，缩放因子不对
                raise ValueError(f'scale {self.scale} is not supported. ' 'Supported scales: 2^n and 3.')

            if self.upsampling == 'Nearest':
                self.postup = nn.Sequential(
                    PA(num_up_feat),
                    nn.GELU(),
                    conv_layer(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm=acb_norm),
                    nn.GELU(),
                    conv_layer(num_up_feat, in_chans, 3, 1, 1, deploy=use_inf, norm=acb_norm)
                )
            else:
                self.postup = nn.Sequential(
                    conv_layer(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm=acb_norm),
                    nn.GELU(),
                    conv_layer(num_up_feat, in_chans, 3, 1, 1, deploy=use_inf, norm=acb_norm)
                )
        
        if self.interpolation == 'PixelShuffle':
            convblock = common.default_conv
            self.lr_up = common.Upsampler(convblock, self.scale, in_chans, act=False) #act='gelu' for v5

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def freeze_patch_emb(self):
    #     self.patch_embed1.requires_grad = False

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        
        # x = self.shallow(x)
        # input = x

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            if i == 0:
                input = x
                if self.down_fea: 
                    x = self.up_fea(x)
                    H = H * 2
                    W = W * 2
            if self.stage_res:
                stage_input = x
            for blk in block:
                x = blk(x)
            if self.stage_norm:
                norm = getattr(self, f"norm{i + 1}")
                x = x.flatten(2).transpose(1, 2)
                # x = x.permute(0, 2, 3, 1).contiguous()
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # x = x.permute(0, 3, 1, 2).contiguous()
            if self.stage_res:
                x = x + stage_input

        if self.down_fea: x = self.down_fea(x)
        x = input + x   # Residual of the whole deep feature Extraction

        return x

    def forward(self, x):
        if hasattr(self, 'sub_mean'):
            x = self.sub_mean(x)
        
        out = self.forward_features(x)

        if self.scale == 1:
            if self.upsampling == 'Nearest' or self.upsampling == 'NearestNoPA':
                out = self.postup(self.preup(self.preup_norm(out)))
            elif self.upsampling == 'PixelShuffle':
                out = self.preup(self.preup_norm(out))
        elif self.upsampling == 'Deconv':
            out = self.deconv(out)
        elif self.upsampling == 'PixelShuffleDirect':
            out = self.pixelshuffledirect(out)
        elif self.upsampling == 'PixelShuffle':
            out = self.preup(self.preup_norm(out))
            out = self.pixelshuffle(out)
        elif self.upsampling == 'Nearest' or self.upsampling == 'NearestNoPA':
            out = self.preup(self.preup_norm(out))
            if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
                for i in range(int(math.log(self.scale, 2))):  #  循环 n 次
                    out = getattr(self, f'up{i}')(F.interpolate(out, scale_factor=2, mode='nearest'))
            elif self.scale == 3:  # 缩放因子等于 3
                out = self.up(F.interpolate(out, scale_factor=3, mode='nearest'))
            out = self.postup(out)

        # a: add interpolate
        # if not self.no_bicubic:
        if self.scale == 1:
            out = out + x
        elif self.interpolation == 'Bicubic':
            out = out + F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        elif self.interpolation == 'Bilinear':
            out = out + F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        elif self.interpolation == 'Nearest':
            out = out + F.interpolate(x, scale_factor=self.scale, mode='nearest')
        elif self.interpolation == 'PixelShuffle':
            out = out + self.lr_up(x)

        if hasattr(self, 'add_mean'):
            out = self.add_mean(out)
        
        return out

