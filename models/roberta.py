from typing import Dict

import torch
from transformers import RobertaTokenizer, RobertaModel

from settings import HUGGING_MODEL_PATH


class RoBERTaTokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = self.roberta_tokenizer = RobertaTokenizer.from_pretrained(
            '{}/codebert-base'.format(HUGGING_MODEL_PATH)
            if HUGGING_MODEL_PATH is not None
            else 'microsoft/codebert-base'
        )

    def forward(self, text_input) -> Dict:
        rlt = self.roberta_tokenizer(
            text_input, return_tensors='pt', padding=True,
            truncation=True, max_length=self.roberta_tokenizer.model_max_length,
        )
        return rlt


class RoBERTaEncoder(torch.nn.Module):
    def __init__(
            self, out_channels: int,
            tuning_layers: int = 0,
            **kwargs,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(
            '{}/codebert-base'.format(HUGGING_MODEL_PATH)
            if HUGGING_MODEL_PATH is not None
            else 'microsoft/codebert-base'
        )

        # freeze weights for tuning layers
        layer_cnt = len(self.roberta.encoder.layer)
        for params in self.roberta.encoder.layer[:layer_cnt - tuning_layers].parameters():
            params.requires_grad = False
        self.out_lin = torch.nn.Linear(768, out_channels)

    def forward(self, input_ids, attention_mask):
        text_features = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.out_lin(text_features.pooler_output)


class RoBERTa(torch.nn.Module):
    def __init__(
            self, out_channels: int,
            tuning_layers: int = 0,
            **kwargs
    ):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.tokenizer = RoBERTaTokenizer()
        self.encoder = RoBERTaEncoder(
            out_channels=out_channels,
            tuning_layers=tuning_layers,
            **kwargs
        ).to(self.device)

    def forward(self, text_input):
        rlt = self.tokenizer(text_input).to(self.device)
        return self.encoder(**rlt)
