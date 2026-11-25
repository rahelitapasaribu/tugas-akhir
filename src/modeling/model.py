"""
Arsitektur IndoBERT untuk klasifikasi sentimen tiga kelas.

Menggunakan pretrained IndoBERT dan Linear Classification Layer.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

MODEL_NAME = "indobenchmark/indobert-base-p1"

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

class IndoBERTClassifier(nn.Module):
    """Model fine-tuning IndoBERT dengan Mean Pooling dan Regularisasi Dropout."""

    def __init__(self, model_name=MODEL_NAME, num_labels=3, dropout_rate=0.25):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate

        self.indobert = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.indobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean Pooling: merata-ratakan seluruh token aktif (bukan hanya [CLS])
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
