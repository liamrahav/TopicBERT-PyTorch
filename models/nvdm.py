'''This module contains the NVDM model.'''
import torch
import torch.nn as nn


class NVDM(nn.Module):
    '''Implementation of the NVDM model as described in `Neural Variational Inference for
    Text Processing (Miao et al. 2016) <https://arxiv.org/pdf/1511.06038.pdf>`_.

    Args:
        vocab_size (int): The vocabulary size that will be used for the BOW's (how long the BOW
            vectors will be).
        num_topics (:obj:`int`, optional): Set to `100` by default. The number of latent topics
            to maintain. Corresponds to hidden vector dimensionality `K` in the technical writing.
        hidden_size(:obj:`int`, optional): Set to `256` by default. The number of hidden units to
            include in each layer of the multilayer perceptron (MLP).
        hidden_layers(:obj:`int`, optional): Set to `1` by default. The number of hidden layers to
            generate when creating the MLP component of the model.
        nonlinearity(:obj:`torch.nn.modules.activation.*`, optional): Set to
            :obj:`torch.nn.modules.activation.Tanh` by default. Controls which nonlinearity to use
            as the activation function in the MLP component of the model.
    '''
    @staticmethod
    def _param_initializer(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, vocab_size, num_topics=100, hidden_size=256, hidden_layers=1, nonlinearity=nn.GELU):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        # First MLP layer compresses from vocab_size to hidden_size
        mlp_layers = [nn.Linear(vocab_size, hidden_size), nonlinearity()]
        # Remaining layers operate in dimension hidden_size
        for _ in range(hidden_layers - 1):
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            mlp_layers.append(nonlinearity())

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.apply(NVDM._param_initializer)

        # Create linear projections for Gaussian params (mean & sigma)
        self.mean = nn.Linear(hidden_size, num_topics)
        self.mean.apply(NVDM._param_initializer)

        # Custom initialization for log_sigma
        self.log_sigma = nn.Linear(hidden_size, num_topics)
        self.log_sigma.bias.data.zero_()
        self.log_sigma.weight.data.fill_(0.)

        self.dec_projection = nn.Linear(num_topics, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, input_bows):
        # Run BOW through MLP
        pi = self.mlp(input_bows)

        # Use this to get mean, log_sig for Gaussian
        mean = self.mean(pi)
        log_sigma = self.log_sigma(pi)

        # Calculate KLD
        kld = -0.5 * torch.sum(1 - torch.square(mean) +
                               (2 * log_sigma - torch.exp(2 * log_sigma)), 1)
        # kld = mask * kld  # mask paddings

        # Use Gaussian reparam. trick to sample from distribution defined by mu, sig
        # This provides a sample h_tm from posterior q(h_tm | V) (tm meaning topic model)
        epsilons = torch.normal(0, 1, size=(
            input_bows.size()[0], self.num_topics)).to(input_bows.device)
        sample = (torch.exp(log_sigma) * epsilons) + mean

        # Softmax to get p(v_i | h_tm), AKA probabilities of words given hidden state
        logits = self.log_softmax(self.dec_projection(sample))

        # Lowerbound on NVDM true loss, used for optimization
        rec_loss = -1 * torch.sum(logits * input_bows, 1)
        loss_nvdm_lb = torch.mean(rec_loss + kld)

        return sample, logits, torch.mean(kld), loss_nvdm_lb
