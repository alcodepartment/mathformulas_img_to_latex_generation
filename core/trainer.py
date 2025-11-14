import os
from typing import Optional
from model import Im2Latex
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

class Trainer:
  """
  Trainer for Im2Latex

  Args:
    model: Im2Latex object
    train_loader: train set
    val_loader: validation set
    pad_id: <pad> id
    device: torch.device
    lr: learning rate
    log_dir: tensorboard logs
    ckpt_dir: checkpoints dir
    ckpt_name: filename for best
    patience: early stopping, epochs wo/ improvement
    resume_from: path from checkpoint to resume, or None
    max_grad_norm: grad clipping norm
    log_interval: how often (in batches) to log within epoch
  """
  def __init__(
      self,
      model: Im2Latex,
      train_loader: DataLoader,
      val_loader: DataLoader,
      pad_id: int,
      device: torch.device,
      lr: float = 1e-3,
      log_dir: str = './runs',
      ckpt_dir: str = './checkpoints',
      ckpt_name: str = 'best.pt',
      patience: int = 5,
      resume_from: Optional[str] = None,
      max_grad_norm: float = 1.0,
      log_interval: int = 50
  ):
    self.model = model.to(device)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.pad_id = pad_id
    self.device = device
    self.lr = lr
    self.log_dir = log_dir
    self.ckpt_dir = ckpt_dir
    self.ckpt_name = ckpt_name
    self.patience = patience
    self.resume_from = resume_from
    self.max_grad_norm = max_grad_norm
    self.log_interval = log_interval

    os.makedirs(self.ckpt_dir, exist_ok = True)
    self.ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)

    self.writer = SummaryWriter(log_dir = log_dir)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      self.optimizer,
      mode = 'min',
      factor = 0.5,
      patience = 2
    )

    self.ce_loss = nn.CrossEntropyLoss(ignore_index = self.pad_id)

    self.start_epoch = 0
    self.best_val_loss = float('inf')
    self.epochs_no_improve = 0
    self.global_step = 0

    if self.resume_from is not None and os.path.isfile(self.resume_from):
      print(f'[Trainer] Resuming from ckpt: {self.resume_from}')
      self.start_epoch, self.best_val_loss = self._load_checkpoint(
        self.resume_from
      )
    else:
      print('[Trainer] Starting from scratch')

  def _save_checkpoint(self, epoch: int):
    os.makedirs(self.ckpt_dir, exist_ok = True)
    torch.save({
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'best_val_loss': self.best_val_loss
    }, self.ckpt_path)
    print(f'[Trainer] Saved ckpt to {self.ckpt_path}')

  def _load_checkpoint(self, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location = self.device)
    self.model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
      self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    print(
      f'[Trainer] Loaded ckpt from {ckpt_path}, '
      f'epoch = {start_epoch}, best_val_loss = {best_val_loss:.2f}'
    )
    return start_epoch, best_val_loss
  
  def _train_one_epoch(self, epoch: int) -> float:
    self.model.train()
    running_loss = 0.0
    num_batches = 0
    total_batches = len(self.train_loader)

    for batch_idx, batch in enumerate(self.train_loader, start=1):
      images = batch['images'].to(self.device)       # (B, C, H, W)
      input_ids = batch['input_ids'].to(self.device)
      target_ids = batch['target_ids'].to(self.device)

      self.optimizer.zero_grad()

      logits, _ = self.model(
        images = images,
        tgt_ids = input_ids,
        mem_pad = None,
        need_xattn = False
      )

      B, T, V = logits.shape
      loss = self.ce_loss(
        logits.view(B * T, V),
        target_ids.view(B * T)
      )

      loss.backward()
      if self.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
          self.model.parameters(), self.max_grad_norm
        )
      self.optimizer.step()

      running_loss += loss.item()
      num_batches += 1

      self.writer.add_scalar(
        'train/loss_step', loss.item(), self.global_step
      )
      self.global_step += 1

      if (
        batch_idx % self.log_interval == 0 or
        batch_idx == 1 or
        batch_idx == total_batches
      ):
        avg_so_far = running_loss / max(1, num_batches)
        print(
          f'  [Train] Epoch {epoch+1} '
          f'Batch {batch_idx}/{total_batches} '
          f'Loss: {loss.item():.4f} '
          f'Avg: {avg_so_far:.4f}'
        )
        self.writer.add_scalar(
          'train/loss_running_avg',
          avg_so_far,
          self.global_step
        )

    epoch_loss = running_loss / max(1, num_batches)
    self.writer.add_scalar('train/loss_epoch', epoch_loss, epoch)

    return epoch_loss
  
  @torch.no_grad()
  def _evaluate(self, epoch: int) -> float:
    self.model.eval()
    running_loss = 0.0
    num_batches = 0
    total_batches = len(self.val_loader)

    for batch_idx, batch in enumerate(self.val_loader, start=1):
      images = batch['images'].to(self.device)       # (B, C, H, W)
      input_ids = batch['input_ids'].to(self.device)
      target_ids = batch['target_ids'].to(self.device)

      logits, _ = self.model(
        images = images,
        tgt_ids = input_ids,
        mem_pad = None,
        need_xattn = False
      )

      B, T, V = logits.shape
      loss = self.ce_loss(
        logits.view(B * T, V),
        target_ids.view(B * T)
      )

      running_loss += loss.item()
      num_batches += 1

      if (
        batch_idx % self.log_interval == 0 or
        batch_idx == total_batches
      ):
        avg_so_far = running_loss / max(1, num_batches)
        print(
          f'  [Val]   Epoch {epoch+1} '
          f'Batch {batch_idx}/{total_batches} '
          f'Loss: {loss.item():.4f} '
          f'Avg: {avg_so_far:.4f}'
        )

    val_loss = running_loss / max(1, num_batches)
    self.writer.add_scalar('val/loss_epoch', val_loss, epoch)
    return val_loss

  def train(self, num_epochs: int = 10):
    print('[Trainer] Starting training')
    for epoch in range(self.start_epoch, num_epochs):
      print(f'\nEpoch {epoch + 1}/{num_epochs}')

      train_loss = self._train_one_epoch(epoch)
      print(f'  train_loss: {train_loss:.4f}')

      val_loss = self._evaluate(epoch)
      print(f'  val_loss:   {val_loss:.4f}')

      self.scheduler.step(val_loss)

      if val_loss < self.best_val_loss:
        print(
          f'  ✓ New best val_loss: {val_loss:.4f} '
          f'(prev {self.best_val_loss:.4f})'
        )
        self.best_val_loss = val_loss
        self.epochs_no_improve = 0
        self._save_checkpoint(epoch)
      else:
        self.epochs_no_improve += 1
        print(
          f'  no improvement for '
          f'{self.epochs_no_improve} epoch(s)'
        )
        if self.epochs_no_improve >= self.patience:
          print(
            f'[Trainer] Early stopping triggered '
            f'after {epoch + 1} epochs'
          )
          break

    self.writer.close()
    print('[Trainer] Training finished')
    print(
      f'[Trainer] Best val_loss: {self.best_val_loss:.4f}, '
      f'best checkpoint: {self.ckpt_path}'
    )
