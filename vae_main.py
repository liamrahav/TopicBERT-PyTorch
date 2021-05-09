import sys

import torch

from datasets import Reuters8Dataset, IMDBDataset, Vocabulary, EmbeddingDataset
from training.pretrain_vae import pretrain
from models import VAEEncoder, Generator2


def run_vae_pretrain(opts):
    verbose = opts['verbose']

    if opts['dataset'] == 'reuters8':
        vocab = Vocabulary.from_files([opts['train_dataset_path'],
                                       opts['val_dataset_path'], opts['test_dataset_path']])
        if verbose:
            print(' [*] Vocabulary built.')

        train_dataset = Reuters8Dataset(
            opts['train_dataset_path'], opts['label_path'], vocab)
        train_dataset = EmbeddingDataset(train_dataset, train_dataset.vocab)
        if verbose:
            print(' [*] Train dataset built.')
        val_dataset = Reuters8Dataset(
            opts['val_dataset_path'], opts['label_path'], vocab)
        val_dataset = EmbeddingDataset(val_dataset, val_dataset.vocab)
        if verbose:
            print(' [*] Validation dataset built.')
        test_dataset = Reuters8Dataset(
            opts['test_dataset_path'], opts['label_path'], vocab)
        test_dataset = EmbeddingDataset(test_dataset, test_dataset.vocab)
        if verbose:
            print(' [*] Test dataset built.')
    elif opts['dataset'] == 'imdb':
        train_dataset, val_dataset, test_dataset = IMDBDataset.full_split(
            opts['train_dataset_path'])
        train_dataset = EmbeddingDataset(train_dataset, train_dataset.vocab)
        if verbose:
            print(' [*] Train dataset built.')
        val_dataset = EmbeddingDataset(val_dataset, val_dataset.vocab)
        if verbose:
            print(' [*] Validation dataset built.')
        test_dataset = EmbeddingDataset(test_dataset, test_dataset.vocab)
        if verbose:
            print(' [*] Test dataset built.')

    tensorboard = not opts['disable_tensorboard']

    load_ckpt = bool(opts['resume'])
    if opts['save_checkpoint_only']:
        load_ckpt = False

    pretrain(train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, emb_size=opts['emb_size'],
             lr=opts['lr'], num_epochs=opts['num_epochs'], num_workers=opts['num_workers'],
             batch_size=opts['batch_size'], device=opts['device'], verbose=verbose,
             tensorboard=tensorboard, tensorboard_dir=opts['tensorboard_dir'], should_load_ckpt=load_ckpt,
             ckpt_dir=opts['resume'])
