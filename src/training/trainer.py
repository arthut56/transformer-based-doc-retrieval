import random
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from src.training.loss import DPRLoss


class PairDataset(Dataset):
    def __init__(self, split):
        self.questions = split['question']
        self.positives = split['text']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return self.questions[i], random.choice(self.positives[i])


class Trainer:
    def __init__(self, dpr, cfg):
        self.dpr = dpr
        self.cfg = cfg
        self.criterion = DPRLoss()
        self.params = list(dpr.query_encoder.parameters()) + list(dpr.document_encoder.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    def train_epoch(self, loader):
        self.dpr.query_encoder.train()
        self.dpr.document_encoder.train()
        total = 0
        for questions, docs in loader:
            q_ids = self.dpr.tokenize(list(questions), self.cfg.model.query_max_length).to(self.dpr.device)
            d_ids = self.dpr.tokenize(list(docs), self.cfg.model.doc_max_length).to(self.dpr.device)

            q_vecs = self.dpr.query_encoder(q_ids)# 64 x 32
            d_vecs = self.dpr.document_encoder(d_ids) #64 x 32
            loss = self.criterion(q_vecs, d_vecs).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()

            total += loss.item()
        return total / len(loader)

    def save(self):
        os.makedirs(self.cfg.train.save_dir, exist_ok=True)
        torch.save(self.dpr.query_encoder.state_dict(), f'{self.cfg.train.save_dir}/query_encoder.pt')
        torch.save(self.dpr.document_encoder.state_dict(), f'{self.cfg.train.save_dir}/document_encoder.pt')