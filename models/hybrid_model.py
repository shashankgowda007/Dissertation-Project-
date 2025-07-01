import torch
import torch.nn as nn
import torchvision.models as models

# Minimal ViT implementation (Base/16) for demonstration
# Since no external dependencies, implement a simple ViT block here

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # B, embed_dim, H/patch, W/patch
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, embed_dim
        x = torch.cat((cls_tokens, x), dim=1)  # B, num_patches+1, embed_dim
        x = x + self.pos_embed
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # B, N, embed_dim
        x = x.transpose(0, 1)    # N, B, embed_dim for MultiheadAttention
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_token_final = x[0]  # CLS token
        cls_token_final = cls_token_final.transpose(0, 1)  # B, embed_dim
        return cls_token_final

class HybridCNNViTModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        # CNN branch: pretrained ResNet-50
        resnet = models.resnet50(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.cnn_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.cnn_feat_dim = 2048

        # ViT branch
        self.vit = VisionTransformer()

        # Fusion layer
        self.fusion_dim = self.cnn_feat_dim + 768  # 2048 + 768 = 2816
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # CNN branch
        cnn_features = self.cnn_backbone(x)  # B, 2048, H, W
        cnn_features = self.cnn_avgpool(cnn_features)  # B, 2048, 1, 1
        cnn_features = torch.flatten(cnn_features, 1)  # B, 2048

        # ViT branch
        vit_features = self.vit(x)  # B, 768

        # Concatenate features
        fused_features = torch.cat((cnn_features, vit_features), dim=1)  # B, 2816

        # Fusion MLP
        fusion_out = self.fusion_mlp(fused_features)  # B, 512

        # Classifier
        logits = self.classifier(fusion_out)  # B, num_classes

        return logits
