import re
import json
from collections import Counter
from typing import List, Dict, Optional

class Tokenizer:
  """
  Tokenizer for OCR LaTeX model

  Breaks LaTeX into structural units (commands, braces, operators, digits, letters)
  Handles special tokens: <pad>, <sos>, <eos>, <unk>.
  Encodes/decodes between strings and integer IDs.
  Saves/loades to/from json
  """

  PATTERN = re.compile(r'(\\[a-zA-Z]+|\\.|[{}^_]|[0-9]+|[A-Za-z]|[^ \t])')

  def __init__(
      self,
      token2id: Optional[Dict[str, int]] = None,
      pad_token: str = '<pad>',
      start_token: str = '<sos>',
      end_token: str = '<eos>',
      unk_token: str = '<unk>'
  ):
    self.pad_token = pad_token
    self.start_token = start_token
    self.end_token = end_token
    self.unk_token = unk_token

    if token2id is None:
      self.token2id = {
        pad_token: 0,
        start_token: 1,
        end_token: 2,
        unk_token: 3,
      }
    else:
      self.token2id = dict(token2id)

    self.id2token = {i: t for t, i in self.token2id.items()}

  def tokenize(self, text: str) -> List[str]:
    """
    Split LaTeX str to list of tokens
    """
    text = text.strip()

    if not text:
      print('No text was found')
      return []
    
    tokens = self.PATTERN.findall(text)
    return tokens
  
  def build_vocab(
      self,
      texts: List[str],
      min_freq: int = 1,
      max_size: Optional[int] = None
  ):
    """
    Build vocab from list of LaTeX strings

    Args:
      texts: list of formulas.
      min_freq: minimum freq for a token to be included.
      max_size: max vocab size (with special tokens).
    """
    counter = Counter()
    for t in texts:
      counter.update(self.tokenize(t))

    sorted_tokens = sorted(
      [tok for tok, c in counter.items() if c >= min_freq],
      key = lambda x: (-counter[x], x)
    )

    if max_size is not None:
      available = max_size - len(self.token2id)
      sorted_tokens = sorted_tokens[:max(0, available)]

    for tok in sorted_tokens:
      if tok not in self.token2id:
        idx = len(self.token2id)
        self.token2id[tok] = idx
        self.id2token[idx] = tok

  def encode(
      self,
      text: str,
      add_sos: bool = True,
      add_eos: bool = True,
  ) -> List[int]:
    """
    Converts str to list of token ids.
    """
    tokens = self.tokenize(text)
    ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]
    if add_sos:
      ids = [self.token2id[self.start_token]] + ids
    if add_eos:
      ids = ids + [self.token2id[self.end_token]]

    return ids
  
  def decode(
      self,
      ids: List[int],
      remove_special: bool = True  
    ) -> str:
    """
    List of tokens to str
    """
    tokens = [self.id2token.get(i, self.unk_token) for i in ids]
    if remove_special:
      specials = {self.pad_token, self.start_token, self.end_token}
      tokens = [t for t in tokens if t not in specials]

    return "".join(tokens)
  
  @property
  def vocab_size(self) -> int:
    return len(self.token2id)
  
  def save(self, path: str):
    """
    Save to json.
    """
    with open(path, 'w', encoding = 'utf-8') as f:
      json.dump(
        {
          "token2id": self.token2id,
          "pad_token": self.pad_token,
          "start_token": self.start_token,
          "end_token": self.end_token,
          "unk_token": self.unk_token
        },
        f,
        ensure_ascii = False,
        indent = 2
      )

  @classmethod
  def load(cls, path: str) -> "Tokenizer":
    """
    Load vocab from json.
    """
    with open(path, 'r', encoding = 'utf-8') as f:
      obj = json.load(f)

    return cls(
      token2id = obj['token2id'],
      pad_token = obj.get('pad_token', '<pad>'),
      start_token = obj.get('start_token', '<sos>'),
      end_token = obj.get('end_token', '<eos>'),
      unk_token = obj.get('unk_token', '<unk>')
    )