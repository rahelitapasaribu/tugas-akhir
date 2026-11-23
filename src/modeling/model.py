
"""
Arsitektur IndoBERT untuk klasifikasi sentimen tiga kelas.

Menggunakan pretrained IndoBERT dan Linear Classification Layer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "indobenchmark/indobert-base-p1"

LABEL2ID = {
    "Negatif": 0,
    "Netral": 1,
    "Positif": 2,
}

ID2LABEL = {
    value: key for key, value in LABEL2ID.items()
}

class IndoBERTClassifier(nn.Module):
    """Model fine-tuning IndoBERT untuk klasifikasi sentimen."""

    def __init__(self, model_name=MODEL_NAME, num_labels=3):
        super().__init__()

        self.indobert = AutoModel.from_pretrained(model_name)

        hidden_size = self.indobert.config.hidden_size

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.indobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Menggunakan representasi token [CLS].
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)

        return logits
