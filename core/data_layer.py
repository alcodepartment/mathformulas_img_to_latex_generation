import os
import csv
from typing import List, Dict, Any, Optional
from tokenizer import Tokenizer

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

class Image2LatexDataset(Dataset):
  """
  Dataset for project.

  Expects csv with file, formula and root dir with imgs
  """
  def __init__(
      self,
      csv_path: str,
      root_dir: str,
      tokenizer: Tokenizer,
      img_height: int = 128,
      transform: Optional[T.Compose] = None
  ):
    self.csv_path = csv_path
    self.root_dir = root_dir
    self.tokenizer = tokenizer
    self.img_height = img_height

    self.samples = []
    with open(csv_path, newline = "", encoding = 'utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
        fname = row['image']
        formula = row['formula']
        img_path = os.path.join(root_dir, fname)
        self.samples.append((img_path, formula))

    if transform is None:
      self.transform = T.Compose(
        [
          T.Grayscale(num_output_channels = 1),
          T.ToTensor(),
          # T.Normalize()
        ]
      )
    else:
      self.transform = transform

  def __len__(self) -> int:
    return len(self.samples)
  
  def _resize_keep_ratio(self, img: Image.Image) -> Image.Image:
    """
    Resize PIL img to fixed height with aspect ratio.
    """
    w, h = img.size
    new_h = self.img_height
    new_w = int(w*(new_h/h))
    if new_w <= 0:
      new_w = 1
    return img.resize((new_w, new_h), Image.BILINEAR)
  
  def __getitem__(self, idx: int) -> Dict[str, Tensor]:
    img_path, formula = self.samples[idx]

    img = self._resize_keep_ratio(Image.open(img_path))
    img = self.transform(img)

    ids = self.tokenizer.encode(formula)

    ids = torch.tensor(ids, dtype = torch.long)

    input_ids = ids[:-1]
    target_ids = ids[1:]

    return {
      "image": img,
      "input_ids": input_ids,
      "target_ids": target_ids,
      "formula": formula
    }
  
def collate_fn(
    batch: List[Dict[str, Tensor]],
    pad_token_id: int
) -> Dict[str, Tensor]:
  """
  Pads sequences to max length in batch
  Pads image width to max width in batch
  """
  images = [item['image'] for item in batch]
  input_ids_list = [item['input_ids'] for item in batch]
  target_ids_list = [item['target_ids'] for item in batch]
  formulas = [item['formula'] for item in batch]

  input_ids = pad_sequence(
    input_ids_list,
    batch_first = True,
    padding_value = pad_token_id
  )

  target_ids = pad_sequence(
    target_ids_list,
    batch_first = True,
    padding_value = pad_token_id
  )

  input_mask = (input_ids != pad_token_id).long()
  target_mask = (target_ids != pad_token_id).long()

  heights = [img.shape[1] for img in images]
  widths = [img.shape[2] for img in images]
  max_w = max(widths)
  H = heights[0]

  padded_images: List[Tensor] = []
  for img in images:
    _, _, w = img.shape
    pad_w = max_w - w
    img_padded = F.pad(img, (0, pad_w, 0, 0), value = 1.0)
    padded_images.append(img_padded)

  images_batch = torch.stack(padded_images, dim = 0) # (B, 1, H, max_w)

  return {
    'images': images_batch,
    'input_ids': input_ids,
    'target_ids': target_ids,
    'input_mask': input_mask,
    'target_mask': target_mask,
    'formulas': formulas
  }

def make_loader(
    csv_path: str,
    root_dir: str,
    tokenizer: Tokenizer,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    img_height: int = 128
) -> DataLoader:
  dataset = Image2LatexDataset(
    csv_path,
    root_dir,
    tokenizer,
    img_height
  )

  pad_id = tokenizer.token2id[tokenizer.pad_token]

  loader = DataLoader(
    dataset,
    batch_size,
    shuffle,
    num_workers = num_workers,
    collate_fn = lambda b: collate_fn(b, pad_id)
  )

  return loader