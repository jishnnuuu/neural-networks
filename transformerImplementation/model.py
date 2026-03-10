import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size : int, embed_size : int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_size)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size : int, seq_len : int, dropout : float) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create matrix of positional encodings of size(seq_len, embed_size)
        pe = torch.zeros(seq_len, embed_size)
        #create a vector of position indices of size(seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #create a vector of dimension indices of size(1, embed_size)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        #calculate the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0)
        #register the positional encodings as a buffer so that they are not updated during training
        self.register_buffer('pe', pe)
        
        def forward(self, x):
            # add the positional encodings to the input embeddings and apply dropout
            x = x + (self.pe[:, :x.shape[1], :]).require_grad_(False)
            x = self.dropout(x)
            return x
            
class LayerNorm(nn.Module):
    def __init__(self, embed_size : int, eps : float = 1e-6) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_size))   #multiplicative parameter
        self.beta = nn.Parameter(torch.zeros(embed_size))   #additive parameter
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, embed_size : int, hidden_size : int, dropout : float) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embed_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size : int, num_heads : int, dropout : float) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "Embed size must be divisible by num heads"
        
        self.query_linear = nn.Linear(embed_size, embed_size) #Wq
        self.key_linear = nn.Linear(embed_size, embed_size) #Wk
        self.value_linear = nn.Linear(embed_size, embed_size) #Wv
        self.out_linear = nn.Linear(embed_size, embed_size) 
        
    @staticmethod    
    def attention(self, query, key, value, mask=None, dropout=nn.Dropout):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        if dropout is not None:
            scores = dropout(scores)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, value)
        return output, attn_weights
    
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # linear projections
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # apply attention
        attn_output, attn_weights = self.attention(query, key, value, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_linear(attn_output)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer_output):
        return x + self.dropout(sublayer_output(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_size : int, num_heads : int, hidden_size : int, dropout : float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForward(embed_size, hidden_size, dropout)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # self attention sublayer
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # feed forward sublayer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers : int, embed_size : int, num_heads : int, hidden_size : int, dropout : float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_size, num_heads, hidden_size, dropout) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_size : int, num_heads : int, hidden_size : int, dropout : float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(embed_size, num_heads, dropout) #cross attention layer
        self.feed_forward = FeedForward(embed_size, hidden_size, dropout)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.norm3 = LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        ''' 
        x is the input to the decoder layer, enc_output is the output from the encoder,
        src_mask is the mask for the encoder-decoder attention, 
        and tgt_mask is the mask for the masked self attention
        '''
        
        # masked self attention sublayer
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # encoder-decoder attention sublayer
        ''' 
        in the encoder-decoder attention, the query comes from the decoder and 
        the key and value come from the encoder 
        '''
        
        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        
        # feed forward sublayer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers : int, embed_size : int, num_heads : int, hidden_size : int, dropout : float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, hidden_size, dropout) for _ in range(num_layers)])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    
class ProjectionLayer(nn.Module):
    def __init__(self, embed_size : int, vocab_size : int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, pos_enc: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pos_enc = pos_enc
        self.proj_layer = proj_layer
        
    def encoder_forward(self, src, src_mask=None):
        src_embedded = self.src_embed(src)
        src_pos_encoded = self.pos_enc(src_embedded)
        enc_output = self.encoder(src_pos_encoded, src_mask)
        return enc_output
    
    def decoder_forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        tgt_embedded = self.tgt_embed(tgt)
        tgt_pos_encoded = self.pos_enc(tgt_embedded)
        dec_output = self.decoder(tgt_pos_encoded, enc_output, src_mask, tgt_mask)
        return dec_output
    
    def projection_forward(self, dec_output):
        return self.proj_layer(dec_output)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, embed_size: int=512, N: int=6, num_heads: int=8, dropout: float=0.1, hidden_size: int=2048) -> Transformer:
    #creating the embedding layers for the source and target vocabularies
    src_embed = InputEmbedding(src_vocab_size, embed_size)
    tgt_embed = InputEmbedding(tgt_vocab_size, embed_size)
    
    #creating the positional encoding layers for the source and target sequences
    src_pos_enc = PositionalEncoding(embed_size, src_seq_len, dropout)
    tgt_pos_enc = PositionalEncoding(embed_size, tgt_seq_len, dropout)
    
    #creating the encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(embed_size, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(embed_size, hidden_size, dropout)
        encoder_block = EncoderBlock(embed_size, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    #creating the decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(embed_size, num_heads, dropout)
        enc_dec_attention_block = MultiHeadAttentionBlock(embed_size, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(embed_size, hidden_size, dropout)
        decoder_block = DecoderBlock(embed_size, decoder_self_attention_block, enc_dec_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
        
    #creating the projection layer to map the decoder output to the target vocabulary
    proj_layer = ProjectionLayer(embed_size, tgt_vocab_size)
    
    #creating the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_enc, tgt_pos_enc, proj_layer)
    
    #initializing the weights of the model using Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

