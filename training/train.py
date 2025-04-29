# Train
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.utils import simulate_missing_modalities

def train(model, dataset, cfg, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0

        for batch in dataloader:
            # 模拟缺失模态
            inputs, labels = batch['modalities'], batch['label']
            inputs, modality_mask = simulate_missing_modalities(inputs, drop_rate=cfg['drop_rate'])

            # 模态预处理
            processed_inputs = {}
            for modality in inputs:
                if modality == 'text':
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    encoded = tokenizer(inputs[modality], return_tensors='pt', padding=True, truncation=True).to(device)
                    processed_inputs[modality] = encoded
                else:
                    processed_inputs[modality] = inputs[modality].to(device)

            outputs = model(processed_inputs, modality_mask)
            labels = labels.to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
