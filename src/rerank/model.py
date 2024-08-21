import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from transformers import AutoModel

class Scorer(nn.Module):
    def __init__(self, bert_model_path, vocab_size, embed_dim=768):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_path)
        self.bert_model.resize_token_embeddings(vocab_size)
        self.ln_score = nn.Linear(embed_dim, 1)

    def forward(self, inputs):
        net = self.bert_model(**inputs)[0]
        net = net[:, 0, :].contiguous()
        score = torch.sigmoid(self.ln_score(nn.functional.relu(net))).squeeze(1)
        return score

    def load_state_dict(self, state_dict, strict=True):
        # Create a new state_dict with keys adjusted to match the model
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = 'bert_model.' + key if not key.startswith('bert_model.') else key
            new_state_dict[new_key] = state_dict[key]

        # Use the adjusted state_dict to load the model weights
        super().load_state_dict(new_state_dict, strict)
 