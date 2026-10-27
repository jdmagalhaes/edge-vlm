#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VisionEncoder(nn.Module):
    def __init__(self, vision_dim=512):
        super().__init__()
        self.model = timm.create_model(
            'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4]
        ) # [B, 80, 32, 32], [B, 160, 16, 16], [B, 960, 8, 8]

        self.proj_early = nn.Conv2d(80, vision_dim, kernel_size=1)
        self.proj_fine = nn.Conv2d(160, vision_dim, kernel_size=1)
        self.proj_coarse = nn.Conv2d(960, vision_dim, kernel_size=1)

    def add_2d_positional_encoding(self, feat):
        B, C, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype

        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)

        yy, xx = torch.meshgrid(y, x, indexing='ij')  # [H, W]

        pos = torch.stack([xx, yy], dim=0)  # [2, H, W]
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)

        return torch.cat([feat, pos], dim=1)


    def forward(self, x):
        features = self.model(x)

        early = features[0]  # [B, 80, 32, 32]
        fine = features[1]  # [B, 160, 16, 16]
        coarse = features[2]  # [B, 960, 8, 8]

        h, w = coarse.shape[2], coarse.shape[3]
        early = F.interpolate(early, size=(h, w), mode='bilinear', align_corners=False)
        fine = F.interpolate(fine, size=(h, w), mode='bilinear', align_corners=False)

        early = self.proj_early(early)  # [B, vision_dim, 8, 8]
        fine = self.proj_fine(fine)  # [B, vision_dim, 8, 8]
        coarse = self.proj_coarse(coarse)  # [B, vision_dim, 8, 8]

        early = self.add_2d_positional_encoding(early)  # [B, vision_dim+2, 8, 8]
        fine = self.add_2d_positional_encoding(fine)  # [B, vision_dim+2, 8, 8]
        coarse = self.add_2d_positional_encoding(coarse)  # [B, vision_dim+2, 8, 8]

        return early, fine, coarse

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=8000, embed_dim=256, hidden_dim=512, output_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 1D convolutions
        self.conv1 = DepthwiseSeparableConv(embed_dim, hidden_dim // 4, kernel_size=(1, 1))
        self.conv3 = DepthwiseSeparableConv(embed_dim, hidden_dim // 4, kernel_size=(1, 3), padding=(0, 1))
        self.conv5 = DepthwiseSeparableConv(embed_dim, hidden_dim // 4, kernel_size=(1, 5), padding=(0, 2))
        self.conv7 = DepthwiseSeparableConv(embed_dim, hidden_dim // 4, kernel_size=(1, 7), padding=(0, 3))

        self.proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.embedding(x)  # [B, seq_len, embed_dim]
        x = x.transpose(1, 2).unsqueeze(2)  # [B, embed_dim, 1, seq_len]

        c1 = F.relu(self.conv1(x))  # [B, hidden_dim//4, 1, seq_len]
        c3 = F.relu(self.conv3(x))  # [B, hidden_dim//4, 1, seq_len]
        c5 = F.relu(self.conv5(x))  # [B, hidden_dim//4, 1, seq_len]
        c7 = F.relu(self.conv7(x))  # [B, hidden_dim//4, 1, seq_len]

        features = torch.cat([c1, c3, c5, c7], dim=1)  # [B, hidden_dim, 1, seq_len]
        features = self.proj(features)  # [B, output_dim, 1, seq_len]

        # For TCN
        token_embeddings = features.squeeze(2).transpose(1, 2)  # [B, seq_len, output_dim]

        # For pooling
        global_embedding = self.global_pool(features).squeeze(2).squeeze(2)  # [B, output_dim]

        return token_embeddings, global_embedding
    
class QuestionConditionedPooling(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        self.query_proj = nn.Linear(dim, num_queries * dim)
        self.key_proj = nn.Linear(dim + 2, dim)
        self.value_proj = nn.Linear(dim + 2, dim)

        self.logit_scale = nn.Parameter(torch.tensor(dim ** -0.5))

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        
        self.attn_drop = nn.Dropout(0.1)
    
    def forward(self, vision_feat, q_global):
        B, _, _, _ = vision_feat.shape

        vision_tokens = vision_feat.flatten(2).transpose(1, 2)  # [B, HW, C+2]

        K = self.k_norm(self.key_proj(vision_tokens))  # [B, HW, C]
        V = self.v_norm(self.value_proj(vision_tokens))  # [B, HW, C]

        Q = self.query_proj(q_global).view(B, self.num_queries, self.dim)
        Q = self.q_norm(Q)  # [B, num_queries, C]

        attn = torch.bmm(Q, K.transpose(1, 2)) * self.logit_scale  # [B, num_queries, HW]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        pooled = torch.bmm(attn, V)  # [B, num_queries, C]
        return pooled + Q

class MultiScaleQuestionPooling(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        self.early_pool = QuestionConditionedPooling(dim, num_queries-12)
        self.fine_pool = QuestionConditionedPooling(dim, num_queries+6)
        self.coarse_pool = QuestionConditionedPooling(dim, num_queries+6)
        # [20, 38, 38] queries for [early, fine, coarse] scales

    def forward(self, vision_feat, q_global):
        early_feat, fine_feat, coarse_feat = vision_feat

        v_early = self.early_pool(early_feat, q_global)
        v_fine = self.fine_pool(fine_feat, q_global)
        v_coarse = self.coarse_pool(coarse_feat, q_global)

        return torch.cat([v_early, v_fine, v_coarse], dim=1)
    
class Chomp2d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :, :-self.chomp]
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=(1, 3), dilation=(1, 1), dropout=0.1):
        super().__init__()
        padding = (kernel_size[1] - 1) * dilation[1]
        padding = (0, padding)

        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.chomp1 = Chomp2d(padding[1])
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        self.chomp2 = Chomp2d(padding[1])
        self.dropout2 = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + x
        
class ConvDecoder(nn.Module):
    def __init__(self, channels=512, num_blocks=7, kernel_size=(1, 3), dropout=0.1):
        super().__init__()
        dilations = [(1, 2 ** i) for i in range(num_blocks)]

        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            for dilation in dilations
        ])

        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output_relu(x)

class DotVLM(nn.Module):
    def __init__(self, question_vocab_size=8000, answer_vocab_size=8000, vision_dim=512, num_queries=32, num_decoder_blocks=7, dropout=0.1):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_queries = num_queries

        self.vision_encoder = VisionEncoder(vision_dim=vision_dim)

        self.text_encoder = TextEncoder(
            vocab_size=question_vocab_size,
            embed_dim=256,
            hidden_dim=512,
            output_dim=vision_dim
        )

        self.pooling = MultiScaleQuestionPooling(dim=vision_dim, num_queries=num_queries)

        self.answer_embed = nn.Embedding(answer_vocab_size, vision_dim, padding_idx=0)

        self.decoder = ConvDecoder(channels=vision_dim, num_blocks=num_decoder_blocks, kernel_size=(1, 3), dropout=dropout)

        self.output_proj = nn.Linear(vision_dim, answer_vocab_size, bias=True)
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        # For better training
        nn.init.normal_(self.answer_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.answer_embed.weight[0].fill_(0.0)
        
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        nn.init.normal_(self.text_encoder.embedding.weight, mean=0.0, std=0.02)

    def encode(self, images, questions):
        vision_features = self.vision_encoder(images)  # (early, fine, coarse): [B, vision_dim+2, 8, 8] each
        
        q_tokens, q_global = self.text_encoder(questions)  # [B, q_len, vision_dim], [B, vision_dim]
        
        visual_tokens = self.pooling(vision_features, q_global)  # [B, 3*num_queries, vision_dim]
        
        return visual_tokens, q_tokens

    def forward(self, images, questions, answers):
        visual_tokens, question_tokens = self.encode(images, questions)

        answer_embeds = self.answer_embed(answers)  # [B, a_len, vision_dim]
        
        full_seq = torch.cat([visual_tokens, question_tokens, answer_embeds], dim=1) # [B, 3*num_queries + q_len + a_len, vision_dim]
        full_seq = full_seq.transpose(1, 2).unsqueeze(2)  # [B, vision_dim, 1, total_len]
        
        decoded = self.decoder(full_seq)  # [B, vision_dim, 1, total_len]
        
        context_len = visual_tokens.shape[1] + question_tokens.shape[1]
        answer_decoded = decoded[:, :, :, context_len:]  # [B, vision_dim, 1, a_len]
        
        answer_decoded = answer_decoded.squeeze(2).transpose(1, 2)  # [B, a_len, vision_dim]
        logits = self.output_proj(answer_decoded)  # [B, a_len, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(self, images, questions, max_length=176, bos_id=2, eos_id=1, temperature=1.0, top_k=50):
        # Autoregressive Token Generation
        B = images.shape[0]
        device = images.device

        visual_tokens, question_tokens = self.encode(images, questions)
        
        context = torch.cat([visual_tokens, question_tokens], dim=1)
        
        # Start with BOS token
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            answer_embeds = self.answer_embed(generated)  # [B, current_len, vision_dim]
            
            full_seq = torch.cat([context, answer_embeds], dim=1)  # [B, context_len + cur_len, vision_dim]
            full_seq = full_seq.transpose(1, 2).unsqueeze(2)
            
            decoded = self.decoder(full_seq)
            
            last_decoded = decoded[:, :, :, -1:]  # [B, vision_dim, 1, 1]
            last_decoded = last_decoded.squeeze(2).transpose(1, 2)  # [B, 1, vision_dim]
            
            logits = self.output_proj(last_decoded)  # [B, 1, vocab_size]
            logits = logits[:, -1, :]  # [B, vocab_size]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = F.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, idx)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_id).all():
                break
        
        # Return without BOS token
        return generated[:, 1:]


if __name__ == "__main__":
    model = DotVLM(
        question_vocab_size=8000,
        answer_vocab_size=8000,
        vision_dim=512,
        num_queries=32,
        num_decoder_blocks=7,
        dropout=0.1
    )
    
    images = torch.randn(1, 3, 256, 256)
    questions = torch.randint(0, 8000, (1, 20))
    answers = torch.randint(0, 8000, (1, 10))
    
    output = model(images, questions, answers)
    print(f"Forward output shape: {output.shape}")

    visual_tokens, question_tokens = model.encode(images, questions)
    print(f"\nVisual tokens: {visual_tokens.shape[1]} tokens")
    print(f"Question tokens: {question_tokens.shape[1]} tokens")
    print(f"Answer tokens: {answers.shape[1]} tokens")
    print(f"Total sequence length: {visual_tokens.shape[1] + question_tokens.shape[1] + answers.shape[1]}")
    
    # For inference
    model.eval()
    generated = model.generate(images, questions, max_length=10, top_k=0)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_trainable_params:,}") # 50,337,715
    print(f"Total parameters: {total_params:,}") # 50,337,715