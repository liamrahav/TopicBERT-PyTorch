'''This module contains the TopicBERT model.'''

import torch
import torch.nn as nn
from transformers import BertModel

from models.nvdm import NVDM


class TopicBERT(nn.Module):
    '''Implementation of the TopicBERT model as described in `TopicBERT for Energy Efficient Document
    Classification (Chaudhary et al. 2020) <https://arxiv.org/pdf/2010.16407.pdf>`_.

    Args:
        vocab_size (int): The vocabulary size to use for the BOW topic model component.
        num_labels (int): The number of labels that there are to classify.
        alpha (:obj:`float`, optional): Defaults to `0.9`. Controls how much to weight
            the cross-entropy loss on top of BERT with the topic model's own loss by the
            following formulation:

            .. math::

                \mathcal{L} = \\alpha\log{p(y=y_l|\mathcal{D})} + (1 -  \\alpha)\mathcal{L}_{\\text{TM}}
        dropout (:obj:`float`, optional): Defaults to `0.1`. Applies the provided `dropout
        <https://jmlr.org/papers/v15/srivastava14a.html>`_ to the joint hidden state :math:`\mathbf{h}_p`.
    '''
    def __init__(self, vocab_size, num_labels, alpha=0.9, dropout=0.1):
        super().__init__()
        self.alpha = alpha
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.nvdm = NVDM(vocab_size)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size + self.nvdm.num_topics,
                      self.encoder.config.hidden_size, bias=False),
            # nn.GELU() # This is NOT used in paper, but it is used in TF source...
            nn.Linear(self.encoder.config.hidden_size, num_labels)
        )
        self.projection.apply(TopicBERT._get_init_transformer(self.encoder))

        self.bert_loss = nn.CrossEntropyLoss(reduction='mean')

    @staticmethod
    def _get_init_transformer(transformer):
        '''Initialization scheme used for transformers:
        https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
        '''
        def init_transformer(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(
                    mean=0.0, std=transformer.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        return init_transformer

    def forward(self, input_ids, attention_mask, bows, labels):
        # (batch_size, length, dim), these are last layer embeddings
        hiddens_last = self.encoder(
            input_ids, attention_mask=attention_mask)[0]
        embs = hiddens_last[:, 0, :]  # [CLS] token embeddings

        h_tm, _, loss_nvdm = self.nvdm(bows)

        # combine hidden states & use/optimize jointly
        logits = self.projection(torch.cat((embs, h_tm), dim=-1))

        # Runs cross-entropy softmax loss on labels
        loss_bert = self.bert_loss(logits, torch.max(labels, 1)[1])
        loss_total = (self.alpha * loss_bert) + ((1 - self.alpha) * loss_nvdm)
        return logits, loss_total
