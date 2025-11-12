# core/model.py
#Никита, я тут пока заглукшу накидал везде

import torch, torch.nn as nn
from .encoder import CNNEncoder
from .decoder import TransformerDecoder
from .tokenizer import stoi, decode

class Image2Latex(nn.Module):
    def __init__(self, vocab_size, d_model=320, heads=4, layers=4, ff=768, drop=0.1):
        super().__init__()
        self.encoder = CNNEncoder(d_model=d_model)
        self.decoder = TransformerDecoder(vocab=vocab_size, d_model=d_model, heads=heads, layers=layers, ff=ff, drop=drop)

    def forward(self, images, tgt_ids):
        mem = self.encoder(images)
        return self.decoder(mem, tgt_ids)

@torch.no_grad()
def greedy_decode(model, img_tensor, max_len=256, bos="<bos>", eos="<eos>"):
    device = next(model.parameters()).device
    model.eval()
    mem = model.encoder(img_tensor.unsqueeze(0).float().to(device))
    seq = torch.tensor([[stoi[bos]]], device=device)
    for _ in range(max_len):
        logits = model.decoder(mem, seq)[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
        if nxt.item()==stoi[eos]: break
    return decode(seq[0].tolist())
