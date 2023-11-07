import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0
        )

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
        )

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size * patch_size * patch_size)
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size * patch_size * patch_size)
        )
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        cube_size,
        patch_size,
        num_heads,
        num_layers,
        dropout,
        extract_layers,
    ):
        super().__init__()
        self.embeddings = Embeddings(
            input_dim, embed_dim, cube_size, patch_size, dropout
        )
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(
                embed_dim, num_heads, dropout, cube_size, patch_size
            )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class MyUNETR(nn.Module):
    def __init__(
        self,
        feature_size=16,
        img_shape=(128, 128, 128),
        input_dim=4,
        output_dim=3,
        embed_dim=768,
        patch_size=16,
        num_heads=12,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = Transformer(
            input_dim,
            embed_dim,
            img_shape,
            patch_size,
            num_heads,
            self.num_layers,
            dropout,
            self.ext_layers,
        )

        self.upsampler = SingleDeconv3DBlock(1, feature_size)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, feature_size * 2, 3),
            Conv3DBlock(feature_size * 2, feature_size * 4, 3),
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim, feature_size * 32),
            Deconv3DBlock(feature_size * 32, feature_size * 16),
            Deconv3DBlock(feature_size * 16, feature_size * 8),
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim, feature_size * 32),
            Deconv3DBlock(feature_size * 32, feature_size * 16),
        )

        self.decoder9 = Deconv3DBlock(embed_dim, feature_size * 32)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, feature_size * 32)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, feature_size * 32),
            Conv3DBlock(feature_size * 32, feature_size * 32),
            Conv3DBlock(feature_size * 32, feature_size * 32),
            SingleDeconv3DBlock(feature_size * 32, feature_size * 16),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 32, feature_size * 16),
            Conv3DBlock(feature_size * 16, feature_size * 16),
            SingleDeconv3DBlock(feature_size * 16, feature_size * 8),
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 16, feature_size * 8),
            Conv3DBlock(feature_size * 8, feature_size * 8),
            SingleDeconv3DBlock(feature_size * 8, feature_size * 4),
        )

        self.decoder0_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 8, feature_size * 4),
            Conv3DBlock(feature_size * 4, feature_size * 4),
            SingleDeconv3DBlock(feature_size * 4, feature_size * 2),
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(feature_size * 3, feature_size * 2),
            Conv3DBlock(feature_size * 2, feature_size * 2),
            SingleConv3DBlock(feature_size * 2, output_dim, 1),
        )

        # self.decoder0_header = \
        #     nn.Sequential(
        #         Conv3DBlock(feature_size * 8, feature_size * 4),
        #         Conv3DBlock(feature_size * 4, feature_size * 4),
        #         SingleConv3DBlock(feature_size * 4, output_dim, 1)
        #     )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        zu = self.upsampler(z0)  # added
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        print("z9", z9.shape)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        print("z6", z6.shape)
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        print("z0", z0.shape)
        z0d = self.decoder0_upsampler(torch.cat([z0, z3], dim=1))
        print("z0d", z0d.shape)
        print("zu", zu.shape)
        print("z0d", z0d.shape)
        output = self.decoder0_header(torch.cat([zu, z0d], dim=1))
        return output


# net = MyUNETR(input_dim=1, output_dim=1)

# # Create a sample input tensor with shape [batch_size, channels, depth, height, width]
# sample_input = torch.randn(1, 1, 128, 128, 128)

# # Pass the sample input through the UNETR model
# output = net(sample_input)

# # Print the shape of the output to see the result
# print("output", output.shape)
# torch.save(net.state_dict(), "model_weights.pth")
# print("net_state_dict", net.state_dict())
# net.load_state_dict(torch.load("model_weights.pth"))