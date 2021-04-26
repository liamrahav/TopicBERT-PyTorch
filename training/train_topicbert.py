'''This module contains code to train & experiment on TopicBERT. All train loops are importable
to be controlled by scripts.

Attributes:
    writer (:obj:`torch.utils.tensorboard.SummaryWriter`): A tensorboard writer used by the module.
        Train loops have a boolean flag to control whether to use it.

'''

import os
import sys
import time

from prefetch_generator import BackgroundGenerator
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from models import TopicBERT

writer = SummaryWriter()


def _configure_optimization(model, num_train_steps, num_warmup_steps, lr, weight_decay=0.01):
    # Copied from: https://huggingface.co/transformers/training.html
    no_decay = ['bias', 'LayerNorm.weight']
    optim_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': weight_decay},
                    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.}]
    optimizer = AdamW(optim_params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_training_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
    return optimizer, scheduler


def train(dataset, batch_size=8, num_warmup_steps=10, lr=2e-5, num_epochs=10, clip=1., device='cpu',
          val_frequency=0., val_dataset=None, test_frequency=0., test_dataset=None, num_workers=8,
          load_ckpt=False, ckpt_dir=None, tensorboard=True, verbose=True, silent=False):
    '''Main training loop for TopicBERT.

    TODO:
        Add logic for val & test datasets.

    TODO:
        Add logic for saving & loading the model from checkpoints

    Args:
        dataset (:obj:`datasets.BOWDataset`): The dataset to train on wrapped as a :obj:`BOWDataset`.
        batch_size (:obj:`int`, optional): Set to :obj:`8` by default. Batch size to feed to the data loader
            for training.
        num_warmup_steps (:obj:`int`, optional): Set to :obj:`10` by default. Number of warmup steps for 🤗
            transformer scheduler.
        lr (:obj:`int`, optional): Set to :obj:`2e-5` by default. Learning rate to be passed to :obj:`AdamW`
            optimizer.
        num_epochs (:obj:`int`, optional): Set to :obj:`10` by default. Number of epochs to train for.
        clip (:obj:`float`, optional): Set to :obj:`1.0` by defualt. Training uses optional gradient clipping
            *by norm*. This means gradients are rescaled with :math:`\\mathbf{g} \\leftarrow \\text{clip}
            \\cdot \\frac{\\mathbf{g}}{\|{\\mathbf{g}}\|}` when :math:`\|{\\mathbf{g}}\| \\geq \\text{clip}`.
        device (:obj:`str`, optional): Set to :obj:`'cpu'` by default. The PyTorch device to use.
        val_frequency (:obj:`float`, optional): Set to :obj:`0.` by default. How frequently in epochs to run
            validation metrics. Must be in range :math:`[0, 1]`. If set to :obj:`1.`, validation metrics will
            be run after 100% of epochs. If set to :obj:`0.5`, validaton metrics will be gathered after 50% of
            epochs (every other epoch), and so on.
        val_dataset (:obj:`datasets.BOWDataset`, optional): Set to :obj:`None` by default. If
            :obj:`val_frequency` > 0, then a validation dataset must be provided.
        test_frequency (:obj:`float`, optional): Set to :obj:`0.` by default. Same as val_frequency, but for
            a test dataset.
        test_dataset (:obj:`datasets.BOWDataset`, optional): Set to :obj:`None` by default. If
            :obj:`test_frequency` > 0, then a validation dataset must be provided.
        num_workers (:obj:`int`, otional): Set to :obj:`8` by default. The number of workers to use for the
            dataloader.
        load_ckpt (:obj:`bool`, optional): Set to :obj:`False` by default. If :obj:`True`, :obj:`ckpt_dir`
            must be provided, and a model will be loaded from :obj:`ckpt_dir`.
        ckpt_dir (:obj:`str`, optional): Set to :obj:`None` by default. If set, after each epoch a copy of the
            model will be kept in :obj:`ckpt_dir`.
        tensorboard (:obj:`bool`, optional): Set to :obj:`True` by default. Whether or not to use PyTorch's
            tensorboard.
        verbose (:obj:`bool`, optional): Set to :obj:`True` by default. Controls verbosity of console output
            when running.
        silent (:obj:`bool`, optional): Set to :obj:`False` by default. If True, nothing will be outputted
            to console.
    '''
    if val_frequency < 0. or val_frequency > 1.:
        raise ValueError(
            'val_frequency has value {}, but must be in [0, 1]'.format(val_frequency))
    if val_frequency > 0. and not val_dataset:
        raise ValueError(
            'val_frequency {} is >0, but no val_dataset has been provided'.format(val_frequency))

    if test_frequency < 0. or test_frequency > 1.:
        raise ValueError(
            'test_frequency has value {}, but must be in [0, 1]'.format(test_frequency))
    if test_frequency > 0. and not test_dataset:
        raise ValueError(
            'test_dataset {} is >0, but no val_dataset has been provided'.format(test_dataset))

    std_out = sys.stdout
    if silent:
        # Send all output to devnull if silent
        f = open(os.devnull, 'w')
        sys.stdout = f

    # TRAIN LOGIC:
    # ============
    model = TopicBERT(len(dataset.vocab), dataset.num_labels).to(device)
    dataloader = dataset.get_dataloader(num_workers=num_workers)
    total_train_steps = len(dataset) // batch_size * num_epochs
    optimizer, scheduler = _configure_optimization(
        model, total_train_steps, num_warmup_steps, lr)

    n_iter = 0
    start_epoch = 0
    # # Load from checkpoint if needed
    # if load_ckpt:
    #     ckpt = load_checkpoint(ckpt_path)
    #     model.load_state_dict(ckpt['net'])
    #     start_epoch = ckpt['epoch']
    #     n_iter = ckpt['n_iter']
    #     optimizer.load_state_dict(ckpt['optim'])
    #     scheduler.load_state_dict(ckpt['sched'])
    #     if verbose:
    #         print(" [*] Finished loading model from checkpoint.")

    loss_avg = float('inf')
    acc_train = 0.
    best_acc_val = 0.
    best_acc_test = 0.

    for epoch in range(start_epoch, num_epochs):
        model.train()
        loss_total = 0.
        num_correct_train = 0

        pbar = tqdm(enumerate(BackgroundGenerator(dataloader)),
                    total=len(dataloader))
        start_time = time.time()

        for batch_ind, (input_ids, attention_mask, bows, labels) in pbar:
            # Move all data to appropriate device
            input_ids = input_ids.to(device).long()
            attention_mask = attention_mask.to(device)
            bows = bows.to(device)
            labels = labels.to(device)
            prepare_time = start_time - time.time()

            # Compute losses, predctions
            logits, loss_batch_total = model(
                input_ids, attention_mask, bows, labels)
            # get label from max score per example
            preds = torch.max(logits, 1)[1]
            num_correct_train += (preds == torch.max(labels, 1)[1]).sum()
            loss_total += loss_batch_total.item()

            loss_batch_avg = loss_batch_total / input_ids.size(0)
            loss_batch_avg.backward()

            if clip > 0.:  # Optional gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()  # optimizer updates model weights based on stored gradients
            scheduler.step()  # Update lr.
            optimizer.zero_grad()  # Reset gradient slots to zero

            process_time = start_time - time.time() - prepare_time
            pbar.set_description(
                f'Compute efficiency: {process_time/(process_time+prepare_time):.2f}, '
                f'batch loss: {loss_batch_avg.item():.2f},  epoch: {epoch + 1}/{num_epochs}')
            start_time = time.time()

            # Update tensorboard with per-batch information
            if tensorboard:
                writer.add_scalar('Loss/train', loss_batch_avg.item(), n_iter)

            n_iter += 1

        # Useful per-epoch training information
        loss_avg = loss_total / len(dataloader.dataset)
        acc_train = num_correct_train / len(dataloader.dataset) * 100.

        # Update tensorboard with per-epoch information
        if tensorboard:
            writer.add_scalar('Accuracy/train', acc_train, epoch + 1)

        if verbose:
            print('Epoch {:3d} | avg loss {:8.4f} | train acc {:2.2f}'.format(
                epoch + 1, loss_avg, acc_train)
            )

        # Save model if ckpt_dir is set every epoch
        # if ckpt_dir:
        #     cpkt = {
        #         'net': model.state_dict(),
        #         'epoch': epoch,
        #         'n_iter': n_iter,
        #         'optim': optimizer.state_dict(),
        #         'sched': scheduler.state_dict()
        #     }
        #     save_checkpoint(cpkt, ckpt_dir)

    if silent:
        # Reset stdout before function exit
        sys.stdout = std_out

    if tensorboard:
        writer.flush()
        writer.close()

    return
