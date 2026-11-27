"""
Arsitektur Fine-Tuned IndoBERT (Base) dengan Weighted Layer Pooling,
Attention + Mean-Max Pooling, dan Multi-Sample Dropout (FP32-Safe Pooling).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

MODEL_NAME = "indobenchmark/indobert-base-p1"

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers=4):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers) / num_hidden_layers)

    def forward(self, all_hidden_states):
        selected_layers = torch.stack(all_hidden_states[-self.num_hidden_layers:], dim=0).float()
        weights = torch.softmax(self.layer_weights.float(), dim=0).view(-1, 1, 1, 1)
        return (weights * selected_layers).sum(dim=0)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence_output, attention_mask):
        scores = self.attn(sequence_output).squeeze(-1).float()
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(sequence_output.float() * weights, dim=1)


class IndoBERTClassifier(nn.Module):
    """Fine-Tuned IndoBERT Classifier dengan Weighted Layer, Attention, & Mean-Max Pooling."""

    def __init__(self, model_name=MODEL_NAME, num_labels=3, dropout_rate=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate

        self.indobert = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        self.layer_pooler = WeightedLayerPooling(num_hidden_layers=4)
        self.attn_pooler = AttentionPooling(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 3)

        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * i) for i in range(1, 6)])
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.indobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = self.layer_pooler(outputs.hidden_states).float()

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        sequence_masked = sequence_output.masked_fill(input_mask_expanded == 0, -1e9)
        max_pooled = torch.max(sequence_masked, dim=1)[0]

        attn_pooled = self.attn_pooler(sequence_output, attention_mask)

        pooled_output = self.layer_norm(torch.cat([mean_pooled, max_pooled, attn_pooled], dim=1))
        logits = sum([self.classifier(dp(pooled_output)) for dp in self.dropouts]) / len(self.dropouts)
        return logits
