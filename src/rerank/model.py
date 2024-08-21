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
        score = torch.sigmoid(self.ln_score(F.relu(net))).squeeze(1)
        return score

    def load_state_dict(self, state_dict, strict=True):
        # Remove the 'bert_model.' prefix from keys in state_dict if present
        new_state_dict = {}
        for key in state_dict.keys():
            if key.startswith('bert_model.'):
                new_key = key[len('bert_model.'):]
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]
        super().load_state_dict(new_state_dict, strict)
 