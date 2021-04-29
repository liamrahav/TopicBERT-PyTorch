'''This module contains code to train & experiment on TopicBERT. All train loops are importable
to be controlled by scripts.

Attributes:
    writer (:obj:`torch.utils.tensorboard.SummaryWriter`): A tensorboard writer used by the module.
        Train loops have a boolean flag to control whether to use it.

'''

import os
import sys
import time

import numpy as np
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import f1_score
import torch
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from models import TopicBERT
from training.utils import save_ckpt, load_ckpt, get_emptiest_gpu


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


def gather_performance(model, t_dataset, batch_size=8, num_workers=8, device='cpu'):
    '''Yields accuracy and F1 score of the provided model for the provided dataset. Performs
    evaluation in eval mode (no dropout, normalization) and does not collect gradients.

    Args:
        model (:obj:`torch.nn.Module`): The model to evaluate
        t_dataset (:obj:`datasets.BOWDataset`): The dataset to evaluate wrapped as a :obj:`BOWDataset`.
        batch_size (:obj:`int`, optional): Set to :obj:`8` by default. Batch size to feed to the data loader
            for evaluation.
        num_workers (:obj:`int`, otional): Set to :obj:`8` by default. The number of workers to use for the
            dataloader.
        device (:obj:`str`, optional): Set to :obj:`'cpu'` by default. The PyTorch device to use.
    '''
    t_dataloader = t_dataset.get_dataloader(
        num_workers=8, batch_size=batch_size, shuffle=False)
    num_correct_val = 0
    best_f1_val = 0
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, bows, labels in t_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            bows = bows.to(device)
            labels = labels.to(device)
            logits, _ = model(input_ids, attention_mask, bows, labels)

            # get label from max score per example
            preds = torch.max(logits, 1)[1]
            labels = torch.max(labels, 1)[1]
            num_correct_val += (preds == labels).sum()

            all_preds = np.concatenate(
                (all_preds, preds.cpu().numpy()), axis=None)
            all_labels = np.concatenate(
                (all_labels, labels.cpu().numpy()), axis=None)

    acc_val = num_correct_val / len(t_dataloader.dataset) * 100.
    return acc_val, f1_score(all_labels, all_preds, average='macro')


def train(dataset, batch_size=8, num_warmup_steps=10, lr=2e-5, alpha=0.9, num_epochs=10, clip=1.,
          device='cpu', val_frequency=0, val_dataset=None, test_frequency=0, test_dataset=None,
          num_workers=8, should_load_ckpt=False, ckpt_dir=None, tensorboard=True, tensorboard_dir=None,
          verbose=True, silent=False):
    '''Main training loop for TopicBERT.

    Args:
        dataset (:obj:`datasets.BOWDataset`): The dataset to train on wrapped as a :obj:`BOWDataset`.
        batch_size (:obj:`int`, optional): Set to :obj:`8` by default. Batch size to feed to the data loader
            for training.
        num_warmup_steps (:obj:`int`, optional): Set to :obj:`10` by default. Number of warmup steps for 🤗
            transformer scheduler.
        lr (:obj:`int`, optional): Set to :obj:`2e-5` by default. Learning rate to be passed to :obj:`AdamW`
            optimizer.
        alpha (:obj:`int`, optional): Defaults to :obj:`0.9`. See :obj:`TopicBERT` for more information.
        num_epochs (:obj:`int`, optional): Set to :obj:`10` by default. Number of epochs to train for.
        clip (:obj:`float`, optional): Set to :obj:`1.0` by defualt. Training uses optional gradient clipping
            *by norm*. This means gradients are rescaled with :math:`\\mathbf{g} \\leftarrow \\text{clip}
            \\cdot \\frac{\\mathbf{g}}{\|{\\mathbf{g}}\|}` when :math:`\|{\\mathbf{g}}\| \\geq \\text{clip}`.
        device (:obj:`str`, optional): Set to :obj:`'cpu'` by default. The PyTorch device to use.
        val_frequency (:obj:`int`, optional): Set to :obj:`0` by default. How frequently in epochs to run
            validation metrics. If set to :obj:`1.`, validation metrics will be run after every epoch. If
            set to :obj:`2`, validaton metrics will be gathered after every other epoch, and so on.
        val_dataset (:obj:`datasets.BOWDataset`, optional): Set to :obj:`None` by default. If
            :obj:`val_frequency` > 0, then a validation dataset must be provided.
        test_frequency (:obj:`int`, optional): Set to :obj:`0` by default. Same as val_frequency, but for
            a test dataset.
        test_dataset (:obj:`datasets.BOWDataset`, optional): Set to :obj:`None` by default. If
            :obj:`test_frequency` > 0, then a validation dataset must be provided.
        num_workers (:obj:`int`, otional): Set to :obj:`8` by default. The number of workers to use for the
            dataloader.
        should_load_ckpt (:obj:`bool`, optional): Set to :obj:`False` by default. If :obj:`True`,
            :obj:`ckpt_dir` must be provided, and a model will be loaded from :obj:`ckpt_dir`.
        ckpt_dir (:obj:`str`, optional): Set to :obj:`None` by default. If set, after each epoch a copy of the
            model will be kept in :obj:`ckpt_dir`.
        tensorboard (:obj:`bool`, optional): Set to :obj:`True` by default. Whether or not to use
            tensorboardx.
        tensorboard_dir (:obj:`str`, optional): Set to :obj:`None` by default. Places tensoboard output in
            the specified directory.
        verbose (:obj:`bool`, optional): Set to :obj:`True` by default. Controls verbosity of console output
            when running.
        silent (:obj:`bool`, optional): Set to :obj:`False` by default. If True, nothing will be outputted
            to console.

    Returns:
        (:obj:`tuple` of :obj:`int`, :obj:`int`): The best validation set and test set accuracy score.
            Note that if either frequency is set to 0, this will yield 0 respectively.
    '''
    if val_frequency > 0. and not val_dataset:
        raise ValueError(
            'val_frequency {} is >0, but no val_dataset has been provided'.format(val_frequency))

    if test_frequency > 0. and not test_dataset:
        raise ValueError(
            'test_dataset {} is >0, but no test_dataset has been provided'.format(test_dataset))

    std_out = sys.stdout
    if silent:
        # Send all output to devnull if silent
        f = open(os.devnull, 'w')
        sys.stdout = f

    writer = None
    if tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard_dir)

    # TRAIN LOGIC:
    # ============
    model = TopicBERT(len(dataset.vocab), dataset.num_labels,
                      alpha=alpha).to(device)
    dataloader = dataset.get_dataloader(
        num_workers=num_workers, batch_size=batch_size, shuffle=True)
    total_train_steps = len(dataset) // batch_size * num_epochs
    optimizer, scheduler = _configure_optimization(
        model, total_train_steps, num_warmup_steps, lr)

    n_iter = 0
    start_epoch = 0

    # Load from checkpoint if needed
    if should_load_ckpt:
        ckpt = load_ckpt(ckpt_dir)
        model.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['sched'])
        if verbose:
            print(" [*] Finished loading model from checkpoint.")

    # Automatically parallelize if >1 GPU available
    if device == 'cuda' and torch.cuda.device_count() > 1:
        torch.cuda.empty_cache()
        model = nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count())))
        # set primary GPU to free-est one
        device = 'cuda:{}'.format(get_emptiest_gpu())

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

        for _, (input_ids, attention_mask, bows, labels) in pbar:
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

            # If parallel, need to manually combine losses across batches
            if 'cuda' in device and torch.cuda.device_count() > 0:
                loss_batch_total = loss_batch_total.sum()

            loss_total += loss_batch_total.item()

            loss_batch_avg = loss_batch_total / (input_ids.size(0))
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
            writer.add_scalar('LossAvg/train', loss_avg, epoch + 1)

        if verbose:
            print('Epoch {:3d} | avg loss {:6.4f} | train acc {:4.2f}'.format(
                epoch + 1, loss_avg, acc_train), end=''
            )

        if (epoch + 1) % val_frequency == 0:
            acc_val, f1_val = gather_performance(model, val_dataset, batch_size=batch_size,
                                                 num_workers=num_workers, device=device)

            if acc_val > best_acc_val:
                best_acc_val = acc_val

            if tensorboard:
                writer.add_scalar('Accuracy/val', acc_val, epoch + 1)
                writer.add_scalar('F1/val', f1_val, epoch + 1)

            if verbose:
                print(' | val acc {:4.2f} | val F1 {:4.2f}'.format(
                    acc_val, f1_val), end='')

        if (epoch + 1) % test_frequency == 0:
            acc_test, f1_test = gather_performance(model, test_dataset, batch_size=batch_size,
                                                   num_workers=num_workers, device=device)

            if acc_test > best_acc_test:
                best_acc_test = acc_test

            if tensorboard:
                writer.add_scalar('Accuracy/test', acc_test, epoch + 1)
                writer.add_scalar('F1/test', f1_test, epoch + 1)

            if verbose:
                print(' | test acc {:4.2f} | test F1 {:4.2f}'.format(
                    acc_test, f1_test), end='')

        print('')

        # Save model if ckpt_dir is set every epoch
        if ckpt_dir:
            cpkt = {
                'net': model.state_dict(),
                'epoch': epoch + 1,
                'n_iter': n_iter,
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict()
            }
            save_ckpt(cpkt, ckpt_dir)

    if silent:
        # Reset stdout before function exit
        sys.stdout = std_out

    if tensorboard:
        writer.close()

    if verbose:
        print('Best\t| val acc {:4.2f} | test acc {:4.2f}'.format(
            best_acc_val, best_acc_test))

    return (best_acc_val, best_acc_test)
