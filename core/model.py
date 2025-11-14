import torch
from torch import nn, Tensor
from typing import Optional, Literal, Tuple
from encoder import ConvEncoder, ConvRowEncoder, ConvEncoder1D
from decoder import LSTMDecoder

class Im2Latex(nn.Module):
  """
  Encoder-decoder for img 2 latex

  Args:
    vocab_size: vocab size..
    encoder_type: 'conv' for ConvEncoder, 'row' for ConvRowEncoder
    d_model: dimension of encoder output and decoder embeds
    hidden_size: lstm hidden size
    attn_size: attn internal size
    in_channels: image channels, 1 in our case (gray)
    dropout: dropout in decoder
  """
  def __init__(
      self,
      vocab_size: int,
      encoder_type: Literal['conv', 'row', 'exp'] = 'conv',
      d_model: int = 256,
      hidden_size: int = 256,
      attn_size: int = 256,
      in_channels: int = 1,
      dropout: float = 0.1
  ):
    super().__init__()

    self.vocab_size = vocab_size
    self.d_model = d_model
    self.hidden_size = hidden_size
    self.encoder_type = encoder_type

    if encoder_type == 'conv':
      self.encoder = ConvEncoder(d_model, in_channels)
    elif encoder_type == 'row':
      self.encoder = ConvRowEncoder(d_model // 2, in_channels)
    elif encoder_type == 'exp':
      self.encoder = ConvEncoder1D(d_model, in_channels)
    else:
      raise ValueError(f'Unk encoder type: {encoder_type}')
    
    assert self.encoder.enc_dim == d_model, (
      f"Encoder out dim {self.encoder.enc_dim} != d_model {d_model}"
    )

    self.decoder = LSTMDecoder(
      vocab_size,
      d_model,
      hidden_size,
      attn_size,
      dropout
    )

  def encode(self, images: Tensor) -> Tensor:
    """
    images: (B, C, H, W)
    out: (B, S, d_model)
    """
    return self.encoder(images)
  
  def forward(
      self,
      images: Tensor, # (B, C, H, W)
      tgt_ids: Tensor, # (B, T) decoder input seq
      mem_pad: Optional[Tensor] = None, # (B, S) True for pad in memory
      need_xattn: bool = False
  ) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Train forward with teacher forcing.
    
    Args:
      images: (B, C, H, W)
      tgt_ids: (B, T) decoder input token ids
      mem_pad: (B, S) mask for enc padding (1 where <pad>)

    Returns:
      logits: (B, T, V)
      attn_seq: (B, T, S) or None
    """
    memory = self.encode(images)
    logits, attn_seq = self.decoder(
      memory = memory,
      tgt_ids = tgt_ids,
      mem_pad = mem_pad,
      need_xattn = need_xattn
    )
    
    return logits, attn_seq
  
  @torch.no_grad()
  def greedy_decode(
    self,
    images: Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int = 512,
    mem_pad: Optional[Tensor] = None
  ) -> Tensor:
    """
    Greedy decode, no teacher forcing

    Args:
      images: (B, C, H, W)
      sos_id: <sos> id
      eos_id <eos> id
      max_len: max decoded tokens
      mem_pad: (B, S) enc mask
    
    Returns:
      decoded_ids: (B, L) of generated token ids
    """
    device = images.device
    memory = self.encode(images)
    B, S, D = memory.shape

    h_t, c_t = self.decoder.init_state(memory, mem_pad)

    coverage = torch.zeros(B, S, device = device, dtype = memory.dtype)
    ctx_prev = torch.zeros(B, D, device = device, dtype = memory.dtype)

    y_t = torch.full(
      (B,),
      fill_value = sos_id,
      dtype = torch.long,
      device = device
    )

    decoded = []
    finished = torch.zeros(B, dtype = torch.bool, device = device)

    for t in range(max_len):
      x_t = self.decoder.emb(y_t) # (B, d_model)
      lstm_in = torch.cat([x_t, ctx_prev], dim = 1) # (B, d_model * 2)

      h_t, c_t = self.decoder.lstm_cell(lstm_in, (h_t, c_t))

      ctx_t, alpha_t = self.decoder.attn(
        h_t, memory, mem_pad = mem_pad, coverage = coverage
      )

      coverage += alpha_t

      dec_feat = torch.cat([h_t, ctx_t], dim = -1) # (B, H + D)
      dec_feat = self.decoder.drop(torch.tanh(self.decoder.fc_ctx(dec_feat))) # (B, d_model)
      logits_t = self.decoder.fc_out(dec_feat) # (B, V)

      # greedy
      next_y = logits_t.argmax(dim = -1) # (B,)

      decoded.append(next_y.unsqueeze(1)) # for all bathces

      ctx_prev = ctx_t
      y_t = next_y

      # track eos
      finished = finished | (next_y == eos_id)
      if finished.all():
        break

    decoded_ids = torch.cat(decoded, dim = 1)
    
    return decoded_ids