import torch
from torch import nn, Tensor

class ConvRowEncoder(nn.Module):
  """
  Conv row BiLSTM features extractor.

  Input: (B, C, H, W)
  Output: (B, S, enc_dim_out) 
  S = H' * W'
  enc_dim_out = 2 * enc_dim
  """
  def __init__(self, enc_dim: int, in_channels = 1):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size = 1, stride = 1),
      nn.ReLU(inplace = True),

      nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),
      nn.MaxPool2d(2, 2), # h/2, w/2

      nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace = True),

      nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),
      nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1)), # h/4, w/2

      nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace = True),
      nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2)), # h/4, w/4

      nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace = True)
    )

    self.row_enc = nn.LSTM(
      input_size = 512,
      hidden_size = enc_dim,
      num_layers = 1,
      batch_first = True,
      bidirectional = True
    )

    self.enc_dim = enc_dim * 2
    

  def forward(self, x: Tensor) -> Tensor:
    """
    images: (B, C, H, W)
    output: (B, S, enc_dim_out), S = H' * W'
    """

    conv_out = self.encoder(x) # (B, 512, H', W')

    B, C, Hp, Wp = conv_out.size()

    conv_out = conv_out.permute(0, 2, 3, 1) # (B, H', W', C)

    lstm_out = []
    for r in range(Hp):
      row_data = conv_out[:, r, :, :] # (B, W', C = 512)
      # lstm needs (B, seq_len, input_size)
      row_out, _ = self.row_enc(row_data) # (B, W', enc_dim * 2)
      lstm_out.append(row_out)

    enc_out = torch.stack(lstm_out, dim = 1) # (B, H', W', enc_dim * 2)
    B, Hp2, Wp2, D = enc_out.size()
    enc_out = enc_out.view(B, Hp2 * Wp2, D) # (B, S, enc_dim * 2)

    return enc_out
  

class ConvEncoder(nn.Module):
  """
  CNN encoder
  Input: (B, C, H, W)
  Output: (B, S, enc_dim) S = H' * W'
  """
  def __init__(self, enc_dim: int, in_channels: int = 1):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),

      nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),

      nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),

      nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),
      nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1)), # h/2 w

      nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(inplace = True),
      nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2)), # h/2 w/2

      nn.Conv2d(512, enc_dim, kernel_size = 3, stride = 1, padding = 1) # (B, enc_dim, H', W')
    )

    self.enc_dim = enc_dim

  def forward(self, x: Tensor) -> Tensor:
    """
    x: (B, C, H, W)
    output: (B, S, enc_dim) S = H' * W'
    """
    feats = self.encoder(x) # (B, enc_dim, H', W')
    B, D, Hp, Wp = feats.size()
    feats = feats.permute(0, 2, 3, 1) # (B, H', W', D)
    enc_out = feats.contiguous().view(B, Hp * Wp, D) # (B, S, enc_dim)

    return enc_out