import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        padding = torch.div((kernel_size - 1), 2, rounding_mode='trunc').item()
        self.block = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,#(kernel_size - 1) // 2,
        )

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True), # TODO: True default
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
            nn.ReLU(inplace=True), # TODO: True default
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
        # self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
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
        in_channels=1,
        out_channels=3,
        embed_dim=768,
        patch_size=16,
        num_heads=12,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = in_channels
        self.output_dim = out_channels
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
            in_channels,
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
            Conv3DBlock(in_channels, feature_size, 3),
            Conv3DBlock(feature_size, feature_size * 2, 3),
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim, feature_size * 16),
            Deconv3DBlock(feature_size * 16, feature_size * 8),
            Deconv3DBlock(feature_size * 8, feature_size * 4),
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim, feature_size * 16),
            Deconv3DBlock(feature_size * 16, feature_size * 8),
        )

        self.decoder9 = Deconv3DBlock(embed_dim, feature_size * 16)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, feature_size * 16)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 32, feature_size * 16),
            Conv3DBlock(feature_size * 16, feature_size * 16),
            Conv3DBlock(feature_size * 16, feature_size * 16),
            SingleDeconv3DBlock(feature_size * 16, feature_size * 8),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 16, feature_size * 8),
            Conv3DBlock(feature_size * 8, feature_size * 8),
            SingleDeconv3DBlock(feature_size * 8, feature_size * 4),
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 8, feature_size * 4),
            Conv3DBlock(feature_size * 4, feature_size * 4),
            SingleDeconv3DBlock(feature_size * 4, feature_size * 2),
        )

        self.decoder0_upsampler = nn.Sequential(
            Conv3DBlock(feature_size * 4, feature_size * 2),
            Conv3DBlock(feature_size * 2, feature_size * 2),
            SingleDeconv3DBlock(feature_size * 2, feature_size),
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(feature_size * 2, feature_size),
            Conv3DBlock(feature_size, feature_size),
            SingleConv3DBlock(feature_size, out_channels, 1),
            
        )

        self.sigmoid = nn.Sigmoid()

        # self.decoder0_header = \
        #     nn.Sequential(
        #         Conv3DBlock(feature_size * 8, feature_size * 4),
        #         Conv3DBlock(feature_size * 4, feature_size * 4),
        #         SingleConv3DBlock(feature_size * 4, output_dim, 1)
        #     )

    def _print_available_memory(self, cuda_device_index=0):
        # Set the device to the desired CUDA device
        torch.cuda.set_device(cuda_device_index)

        # Get total and allocated memory in bytes
        total_memory = torch.cuda.get_device_properties(cuda_device_index).total_memory
        allocated_memory = torch.cuda.memory_allocated(cuda_device_index)

        # Calculate free memory
        free_memory = total_memory - allocated_memory

        # Convert bytes to gigabytes for easier interpretation
        total_memory_gb = total_memory / (1024**3)
        allocated_memory_gb = allocated_memory / (1024**3)
        free_memory_gb = free_memory / (1024**3)
        print("--------------------------------------")
        print(f"Total VRAM: {total_memory_gb:.2f} GB")
        print(f"Allocated VRAM: {allocated_memory_gb:.2f} GB")
        print(f"Free VRAM: {free_memory_gb:.2f} GB")
        print("--------------------------------------")

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        # print("input z0 zeros:", torch.sum(z0.eq(0)), torch.sum(z0 == 0) / z0.numel())
        # print("z3 zeros:", torch.sum(z3.eq(0)), torch.sum(z3 == 0) / z3.numel())
        # print("z6 zeros:", torch.sum(z6.eq(0)), torch.sum(z6 == 0) / z6.numel())
        # print("z9 zeros:", torch.sum(z9.eq(0)), torch.sum(z9 == 0) / z9.numel())
        # print("z12 zeros:", torch.sum(z12.eq(0)), torch.sum(z12 == 0) / z12.numel())


        zu = self.upsampler(z0)  # added
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        # print("z0 zeros:", torch.sum(z0.eq(0)), torch.sum(z0 == 0) / z0.numel())
        z0d = self.decoder0_upsampler(torch.cat([z0, z3], dim=1))

        # Set the device to the desired CUDA device

        # print the mean of z0d
        # print("zu zeros:", torch.sum(zu.eq(0)), torch.sum(zu == 0) / zu.numel())
        # print("z0d zeros:", torch.sum(z0d.eq(0)), torch.sum(z0d == 0) / z0d.numel())
        logits = self.decoder0_header(torch.cat([zu, z0d], dim=1))

        # TODO: remove
        # all_logits_np = logits.cpu().detach().numpy()
        # # Flatten the array if it's multi-dimensional
        # all_logits_np = all_logits_np.flatten()

        # # Plotting the distribution of logits
        # plt.hist(all_logits_np, bins=50)  # You can adjust the number of bins
        # plt.xlabel('Logit value')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Logits')
        # # random number between 1 and 100
        # rand_num = np.random.randint(1, 20)
        # plt.savefig(f"logits_distribution{rand_num}.png")

        output = self.sigmoid(logits)
        
        # if torch.isnan(logits).any():
        #     # print 10 random values of logits
        #     print("logits", logits[torch.randperm(logits.size()[0])[:10]])

        # print("output zeros:", torch.sum(output.eq(0)), torch.sum(output == 0) / output.numel())
        # torch.cuda.empty_cache()

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
