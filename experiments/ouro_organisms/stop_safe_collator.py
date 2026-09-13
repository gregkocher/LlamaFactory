"""Unwired prototype for a separate native-PT recipe with genuine EOS supervision.

The active Ouro trainer does not import this module. This preserves input tokens,
packing, attention, and the model's own next-token shift/loss. It changes only
which target positions are ignored. Packed document boundaries remain EOS tokens;
this does not introduce document-isolated attention or change cross-document loss.
"""
from dataclasses import dataclass
from typing import Any

import torch


def labels_from_attention(input_ids, attention_mask):
    """Mask padding by position, never by token identity (pad may equal EOS).

A real token immediately after left padding is also ignored: its predecessor is
padding, not an observed context token. Position zero is left to the native causal
loss shift, which excludes it. Internal attended EOS and following document starts
remain supervised, including when EOS has the same token ID as padding.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('Require matching [batch, sequence] tensors')
    if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
        raise ValueError('Require a binary attention mask; segment-ID masks are a different recipe')
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, 1:][attention_mask[:, :-1] == 0] = -100
    return labels


@dataclass
class AttentionMaskCausalLMCollator:
    """Prototype drop-in causal-LM collator; integration requires a new recipe."""
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        if not features or any('labels' in row for row in features):
            raise ValueError('Require nonempty native-PT tokenized features without preexisting labels')
        batch = self.tokenizer.pad(features, padding=True,
                                   pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt')
        batch.pop('special_tokens_mask', None)
        batch.pop('offset_mapping', None)
        if 'attention_mask' not in batch:
            raise ValueError('An explicit padding attention mask is required')
        batch['labels'] = labels_from_attention(batch['input_ids'], batch['attention_mask'])
        return batch
