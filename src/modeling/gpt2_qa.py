import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .gpt2_patch import GPT2QuantModel 

class GPT2QuantForQA(nn.Module):
    def __init__(self, model_name: str, profile: Dict[str, Any], dropout: float = 0.1):
        super().__init__()
        self.backbone = GPT2QuantModel(model_name, profile)
        hidden = self.backbone.model.config.n_embd
        self.qa_dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(hidden, 2) 

    @torch.no_grad()
    def set_bits_profile(self, profile: dict):
        self.backbone.set_bits_profile(profile)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs = self.backbone.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  
        x = self.qa_dropout(hidden_states)
        logits = self.qa_outputs(x)               
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)   
        end_logits = end_logits.squeeze(-1)       

        loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = (loss_fct(start_logits, start_positions) +
                    loss_fct(end_logits, end_positions)) / 2.0

        return QuestionAnsweringModelOutput(
            loss=loss, start_logits=start_logits, end_logits=end_logits
        )
