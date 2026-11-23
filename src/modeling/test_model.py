
"""
Menguji pemuatan model dan bentuk output IndoBERT.
"""

import torch

from model import IndoBERTClassifier

def main():
    model = IndoBERTClassifier()
    model.eval()

    batch_size = 2
    sequence_length = 128

    input_ids = torch.randint(
        low=0,
        high=1000,
        size=(batch_size, sequence_length)
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long
    )

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = torch.softmax(logits, dim=1)

    print("Model berhasil dimuat.")
    print("Logits shape:", logits.shape)
    print("Probabilities shape:", probabilities.shape)
    print("Probabilities:")
    print(probabilities)

if __name__ == "__main__":
    main()
