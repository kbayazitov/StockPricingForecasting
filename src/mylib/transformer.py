from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return token_embedding + self.pos_embedding[:token_embedding.size(0),:]

class TokenEmbedding(nn.Module):
    def __init__(self, feature_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Linear(feature_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens)

class Seq2SeqTransformer(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, feature_size, emb_size, hidden_dim, num_layers, nhead, trg_len=5):
        super(Seq2SeqTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.src_tok_emb = TokenEmbedding(feature_size, emb_size)
        self.trg_tok_emb = TokenEmbedding(1, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.generator = nn.Linear(emb_size, 1)
        self.trg_len = trg_len

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg=None):

        src = src.transpose(1, 0) # -> [src_len, batch_size, feature_size]
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer_encoder(src_emb)

        if trg is not None:
            trg = trg.transpose(1, 0) # -> [trg_len, batch_size, feature_size]
            trg_mask = generate_square_subsequent_mask(trg.shape[0])
            trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
            outs = self.transformer_decoder(trg_emb, memory, trg_mask)
            return self.generator(outs).transpose(1, 0).squeeze()

        else:
            outputs = torch.zeros(src.size(1), self.trg_len, 1).to(self.device) # [batch_size, target_len, input_size]
            #trg = src[-1, :, 3].unsqueeze(0).unsqueeze(2)
            trg = torch.zeros(1, src.size(1), 1).to(self.device)

            for t in range(self.trg_len):
                trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
                trg_mask = generate_square_subsequent_mask(trg_emb.shape[0])
                #out = self.transformer_decoder(trg_emb, memory, trg_mask)
                out = self.transformer_decoder(trg_emb, memory)
                out = self.generator(out)
                outputs[:, t:t+1] = out[-1:, :, :].transpose(1, 0)
                trg = torch.cat((trg, out[-1:, :, :]), dim=0)

            return outputs.squeeze(2)
