# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)

from timm.models.layers import trunc_normal_
from .convnextv2_sparse import SparseConvNeXtV2
from .convnextv2 import Block
from .norm_layers import LayerNorm

class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
                self,
                img_size=512,
                in_chans=3,
                depths=[3, 3, 9, 3],
                dims=[96, 192, 384, 768],
                decoder_depth=1,
                decoder_embed_dim=512,
                patch_size=32,
                mask_ratio=0.6,
                norm_pix_loss=False,
                args=None):
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.args = args
        # encoder
        self.encoder = SparseConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims, D=2)
        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.decoder_dict = nn.ModuleDict()
        self.pred_dict = nn.ModuleDict()
        # for this task, we create 2 predictions, one for the image, and one for the domain which is a classification task
        self.decoder_dict["image"] = nn.Sequential(*decoder)
        self.pred_dict["image"] = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)
        
        if args.domain_task:
            self.decoder_dict["domain"] = nn.Sequential(*decoder)
            self.layer_norm_tmp = LayerNorm(
                decoder_embed_dim, eps=1e-6, data_format="channels_first"
            )
            self.pred_dict["domain"] = nn.Linear(
                in_features=decoder_embed_dim,
                out_features=9,
            )
        if args.edge_task:
            self.decoder_dict["edges"] = nn.Sequential(*decoder)
            self.pred_dict["edges"] = nn.Conv2d(
                in_channels=decoder_embed_dim,
                out_channels=patch_size ** 2,
                kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs, modality = None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        if modality == "edges":
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    
    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):
        pred = {}
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask

        x_ = self.decoder_dict["image"](x)
        pred["image"] = self.pred_dict["image"](x_)

        if self.args.domain_task:
            x_ = self.decoder_dict["domain"](x)
            x_ = self.layer_norm_tmp(x_)
            x_ = x_.mean(dim=[-2, -1])
            pred["domain"] = self.pred_dict["domain"](x_)

        if self.args.edge_task:
            x_ = self.decoder_dict["edges"](x)
            pred["edges"] = self.pred_dict["edges"](x_)

        return pred

    def forward_loss(self, imgs, pred, mask, domain_labels, edge_target):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        loss_dict = {}
        if len(pred["image"].shape) == 4:
            n, c, _, _ = pred["image"].shape
            pred["image"] = pred["image"].reshape(n, c, -1)
            pred["image"] = torch.einsum('ncl->nlc', pred["image"])

        if self.args.edge_task:
            if len(pred["edges"].shape) == 4:
                n, c, _, _ = pred["edges"].shape
                pred["edges"] = pred["edges"].reshape(n, c, -1)
                pred["edges"] = torch.einsum('ncl->nlc', pred["edges"])

        target = self.patchify(imgs)    
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss_dict["image"] = (pred["image"] - target) ** 2
        loss_dict["image"] = loss_dict["image"].mean(dim=-1)  # [N, L], mean loss per patch
        loss_dict["image"] = (loss_dict["image"] * mask).sum() / mask.sum()  # mean loss on removed patches

        if self.args.domain_task:
            loss_dict["domain"] = nn.CrossEntropyLoss()(pred["domain"], domain_labels)

        if self.args.edge_task:
        # pred is of shape 128, 256, 1024
            pred_edges = pred["edges"].unsqueeze(-1) # 128, 256, 1024, 1
            edge_target = self.patchify(edge_target, "edges")
            edge_pred = torch.sigmoid(pred["edges"]) # 128, 256, 1024
            edge_mask = (
                        mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2).unsqueeze(-1)
                    ).squeeze(-1) # 128, 256, 1024
            loss_dict["edges"] = nn.BCELoss()(edge_pred * edge_mask, edge_target * edge_mask)


        return loss_dict

    def forward(self, imgs, domain_labels=None, edge_target=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss_dict = self.forward_loss(imgs, pred, mask, domain_labels, edge_target)
        loss = loss_dict["image"]
        if self.args.domain_task:
            if self.args.inverse_domain_task:
                loss -= loss_dict["domain"] # inverse domain task 
            else:
                loss += loss_dict["domain"]
        if self.args.edge_task:
            loss += loss_dict["edges"]
        return loss, loss_dict, pred, mask

def convnextv2_atto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = FCMAE(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = FCMAE(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model