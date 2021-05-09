'''The holistic regularisation VAE from Li et al. 2019: https://arxiv.org/pdf/1911.05343.pdf.
This module contains only the

Copyright notice:

MIT License

Copyright (c) 2019 Ruizhe Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''


import random

import torch
from torch import nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.layer_dim = (n_layers*2 if bidirectional else n_layers)*hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        self.linear_mu = nn.Linear(self.layer_dim*2, self.layer_dim*2)
        self.linear_var = nn.Linear(self.layer_dim*2, self.layer_dim*2)


    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def get_sen_len(self, sens):
        length = torch.sum(sens > 0, dim=0)
        return length.to(dtype=torch.float)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim]
        mu_ = []
        logvar_ = []
        hx = None
        for i in range(embedded.shape[0]):

            _, hx = self.rnn(embedded[i].unsqueeze(0), hx)
            h = self.ziphidden(*hx)
            # cat hidden and cell at each time stamp
            mu = self.linear_mu(h)
            logvar = self.linear_var(h)
            h = self.reparameterize(mu, logvar)  # z = h
            mu_.append(mu)
            logvar_.append(logvar)
        # outputs = [src sent len, batch size, hid dim * n directions]
        # outputs are always from the top hidden layer


        mu = torch.stack(mu_)
        logvar = torch.stack(logvar_)


        return h, mu, logvar, self.get_sen_len(src)

    def ziphidden(self, hidden, cell):
        b_size = hidden.shape[1]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        h = torch.cat([hidden, cell], dim=2)
        # h = [n layers * n directions, batch size, hid dim * 2]
        h = torch.transpose(h, 0, 1).contiguous()
        # h = [batch size, n layers * n directions, hid dim * 2]



        h = h.view(b_size, -1)
        # h = [batch size, n layers * n directions * hid dim * 2]
        return h

    def loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / mu.shape[0]

        return KLD


class Generator2(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=False, teacher_force=0.5):
        super(Generator2, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.layer_dim = n_layers*2 if bidirectional else n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)

        self.rnn = nn.LSTM(emb_dim+hid_dim*2, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)

        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.teacher_force = teacher_force

    def singledecode(self, input, hidden, cell, lat_z=None):
        # first input to the decoder is the <sos> tokens
        input = input.unsqueeze(0)
        if lat_z.type() != None:
            lat_z = lat_z.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]


        emba_cat = torch.cat([embedded,lat_z], dim=2)


        output, (hidden, cell) = self.rnn(emba_cat, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

    def forward(self, input, sen):
        b_size = input.shape[0]
        zz = input.view(b_size, self.layer_dim, -1)
        # zz = [batch size, n layers * n directions, hid dim * 2]
        zzz = torch.transpose(zz, 0, 1)
        # zzz = [n layers * n directions, batch size, hid dim * 2]

        hidden = zzz[:, :, :self.hid_dim].contiguous()
        cell = zzz[:, :, self.hid_dim:].contiguous()
        # cell = [n layers * n directions, batch size, hid dim]

        # hidden = [n layers * n directions, batch size, hid dim]

        max_len = sen.shape[0] #if self.training else sen

        outputs = []
        input = torch.tensor([2]*b_size, device=input.device)
        for t in range(1, max_len):
            output, hidden, cell = self.singledecode(input, hidden, cell, lat_z=zzz[-1,:,:])
            outputs.append(output)
            teacher_force = random.random() < self.teacher_force
            top1 = output.max(1)[1]
            input = torch.tensor([1]*b_size, device=input.device)

        output = torch.stack(outputs)
        return output

    def loss(self, prod, target, weight):
        # prod = torch.softmax(prod, 2)
        recon_loss = F.cross_entropy(
            prod.contiguous().view(-1, prod.shape[2]), target[1:].contiguous().view(-1),
            ignore_index=0, reduction="sum")
        return recon_loss
