import os
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from transformers import TrainingArguments
import numpy as np
import torchmetrics

from torch import optim
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer
# from transformers import get_schedulers
# from datasets import load_dataset

from model.model import UserEmbedding, RankingScore
from utils.helper import RecDataset

import yaml
with open('./config.yaml') as f:
    cfg = yaml.safe_load(f)

n_candidate = cfg['dataset']['args']['cand_dim']
num_epochs = cfg['train']['num_epochs']
bz_train = cfg['train']['bz_train']
bz_val = cfg['train']['bz_val']
os.makedirs(cfg['ckp_dir'], exist_ok=True)

def cat_nce(inputs):
    batch = [[], [], [], [], []]
    for item in inputs:
        for i in range(n_candidate):
            batch[i].extend(item[i])
    for i in range(n_candidate):
        batch[i] = torch.tensor(batch[i])
    return batch

tokenizer= AutoTokenizer.from_pretrained(cfg['model']['pretrain'])
dataset = RecDataset(tokenizer, **cfg['dataset']['arg'])
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bz_train, collate_fn=cat_nce, drop_last=True)    
val_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=bz_val, collate_fn=cat_nce, drop_last=True)

device = cfg['train']['device']
model = RankingScore(cfg['model']['pretrain'], cfg['model']['hidden'], only_feature=True)
model.to(device=device)

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=num_epochs, pct_start=0.1)

best = 0.0
entropy = torch.nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    print(f'--- {epoch} / {num_epochs} ---')
    model.train()
    for batch in tqdm(train_dataloader):
        hist, mask_hist, cand, cand_hist, labels = (b.to(device = device) for b in batch)
        outputs = model(hist, mask_hist, cand, cand_hist, bz_train)
        outputs = outputs.squeeze(-1)
        loss = entropy(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        metric = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        for batch in tqdm(val_dataloader):
            hist, mask_hist, cand, cand_hist, labels = (b.to(device = device) for b in batch)
            outputs = model(hist, mask_hist, cand, cand_hist, bz_val)
            outputs = outputs.squeeze(-1)

            pred = torch.argmax(outputs, axis=-1)
            metric(pred.cpu(), labels.cpu())
        acc = metric.compute().item()
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(cfg['ckp_dir'], f'best_{best}.pt'))
            print("save best: ", best)
    scheduler.step()
