import math
import random
import time

from prefetch_generator import BackgroundGenerator
import torch
from torch import nn
from tqdm import tqdm

from models import VAEEncoder, Generator2
from training.utils import save_ckpt, load_ckpt, get_emptiest_gpu


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def sentence_acc(prod, target):
    target = target[1:]
    mask = target == 0
    prod = prod.argmax(dim=2)
    prod[mask] = -1
    correct = torch.eq(prod, target).to(dtype=torch.float).sum()
    return correct.item()


def reconstruction(encoder, decoder, dataloder, device='cuda', verbose=True):
    encoder.eval()
    decoder.eval()

    recon_loss_total = 0
    vae_loss_total = 0
    words_total = 0
    batch_total = 0
    for i, sen in enumerate(dataloder):
        # sen = sen.permute(1, 0)
        b_size = sen.shape[1]
        if 'cuda' in device:
            sen = sen.cuda(device=0)
        else:
            sen = sen.to(device)
        with torch.no_grad():
            z, mu, logvar, sen_len = encoder(sen)

            recon_mu = decoder(z, sen)
            vae_loss = None
            recon_loss = None
            if isinstance(encoder, nn.DataParallel):
                vae_loss = encoder.module.loss(mu, logvar)
                recon_loss = decoder.module.loss(recon_mu, sen, sen_len)
            else:
                vae_loss = encoder.loss(mu, logvar)
                recon_loss = decoder.loss(recon_mu, sen, sen_len)
            # recon = [sen len, batch size, vocab size]

            sens_mu = recon_mu.argmax(dim=2)

            recon_loss_total = recon_loss_total + recon_loss.item()
            vae_loss_total = vae_loss_total + vae_loss.item()

            words = sen_len.sum().item()

            words_total = words_total + words
            batch_total += b_size

    if verbose:
        print(f"Eval: words_total:{words_total}, batch_total:{batch_total}, recon_loss:{(recon_loss_total/(batch_total)):.04f}, kl_loss:{(vae_loss_total/(batch_total)):.04f}, nll_loss:{((recon_loss_total+vae_loss_total)/(batch_total)):.04f}, ppl:{(math.exp((recon_loss_total+vae_loss_total)/words_total)):.04f}")
    return math.exp((recon_loss_total+vae_loss_total)/words_total)


def _run_epoch(encoder, decoder, opt, dataloader, epoch, device='cuda', verbose=True):
    print("--------------------------")
    encoder.train()
    decoder.train()
    total = 0
    recon_loss_total = 0
    vae_loss_total = 0
    mini_loss_total = 0
    correct_total = 0
    words_total = 0
    batch_total = 0

    # pbar = tqdm(,
    #             total=len(dataloader))

    start_time = time.time()
    for i, sen in enumerate(BackgroundGenerator(dataloader)):
        # sen = sen.permute(1, 0)
        # sen: [len_sen, batch]
        batch_size = sen.shape[1]
        opt.zero_grad()
        total += sen.shape[1]
        if 'cuda' in device:
            sen = sen.cuda(device=0)
        else:
            sen = sen.to(device)
        z, mu, logvar, sen_len = encoder(sen)

        prod = decoder(z, sen)
        vae_loss = None
        recon_loss = None
        if isinstance(encoder, nn.DataParallel):
            vae_loss = encoder.module.loss(mu, logvar)
            recon_loss = decoder.module.loss(prod, sen, sen_len)
        else:
            vae_loss = encoder.loss(mu, logvar)
            recon_loss = decoder.loss(prod, sen, sen_len)

        ((vae_loss+recon_loss)*1).backward()
        opt.step()

        # pbar.set_description(desc='Epoch: {}'.format(epoch))

        recon_loss_total += recon_loss.item()
        vae_loss_total += vae_loss.item()

        correct = sentence_acc(prod, sen)
        words = sen_len.sum().item()
        correct_total = correct_total + correct
        words_total = words_total + words
        batch_total += batch_size

        if i % 10 == 0:
            print('.', end='')
    print('')

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(
        f"Epoch {epoch} | Time: {epoch_mins}m {epoch_secs}s | recon_loss={(recon_loss_total/(batch_total)):.04f}, kl_loss={(vae_loss_total/(batch_total)):.04f}, nll_loss={((recon_loss_total+vae_loss_total)/(batch_total)):.04f}, ppl={(math.exp((recon_loss_total+vae_loss_total)/words_total)):.04f}, acc={(correct_total/words_total):.04f}")
    return recon_loss_total/(batch_total), vae_loss_total/(batch_total), correct_total/words_total, (recon_loss_total+vae_loss_total)/(batch_total), math.exp((recon_loss_total+vae_loss_total)/words_total), end_time - start_time


def pretrain(dataset, val_dataset=None, test_dataset=None, emb_size=512, hidden_size=256, lr=0.0001, num_epochs=1000, num_workers=8, batch_size=128, device='cuda', verbose=True, tensorboard=True, tensorboard_dir='runs', should_load_ckpt=False, ckpt_dir=None):
    writer = None
    if tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard_dir)

    dataloader = dataset.get_dataloader(
        num_workers=num_workers, batch_size=batch_size, shuffle=True)

    val_dataloader = None
    if val_dataset:
        val_dataloader = val_dataset.get_dataloader(
            num_workers=num_workers, batch_size=batch_size, shuffle=False)

    test_dataloader = None
    if test_dataset:
        test_dataloader = test_dataset.get_dataloader(
            num_workers=num_workers, batch_size=batch_size, shuffle=False)

    vocab_size = len(dataset.vocab)
    encoder = VAEEncoder(vocab_size, emb_size, hidden_size, 2, 0.5).to(device)
    decoder = Generator2(vocab_size, emb_size, hidden_size, 2).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) +
                                 list(decoder.parameters()), lr=lr, eps=1e-6, weight_decay=1e-5)
    start_epoch = 0

    if should_load_ckpt:
        ckpt = load_ckpt(ckpt_dir)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optim'])
        if verbose:
            print(" [*] Finished loading model from checkpoint.")

    # Automatically parallelize if >1 GPU available
    if 'cuda' in device and torch.cuda.device_count() > 1:
        torch.cuda.empty_cache()
        encoder = nn.DataParallel(encoder, device_ids=list(
            range(torch.cuda.device_count())), dim=1)
        decoder = nn.DataParallel(decoder, device_ids=list(
            range(torch.cuda.device_count())), dim=1)
        # set primary GPU to free-est one
        device = 'cuda:{}'.format(get_emptiest_gpu())
        if verbose:
            print(' [*] Parallel Mode: Using {} GPUs. {} primary device.'.format(
                torch.cuda.device_count(), device))

    best_ppl_val = 1e9
    best_ppl_test = 1e9
    for epoch in range(start_epoch, num_epochs):
        recon_loss, vae_loss, acc, nll_loss, ppl, runtime = _run_epoch(
            encoder, decoder, optimizer, dataloader, epoch, device=device, verbose=verbose)
        if tensorboard:
            writer.add_scalar('EpochTime/train', runtime, epoch)
            writer.add_scalar('Perplexity/train', ppl, epoch)
            writer.add_scalar('NLL_Loss/train', nll_loss, epoch)
            writer.add_scalar('KLD/train', vae_loss, epoch)
            writer.add_scalar('Recon_Loss/train', recon_loss, epoch)
            writer.add_scalar('ELBO/train', vae_loss+recon_loss, epoch)

        val_ppl = 1
        if val_dataloader:
            val_ppl = reconstruction(
                encoder, decoder, val_dataloader, verbose=verbose)
            if tensorboard:
                writer.add_scalar('Perplexity/val', val_ppl, epoch)

        # Save model if ckpt_dir is set & performance is best
        if ckpt_dir and (val_ppl < best_ppl_val):
            cpkt = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'epoch': epoch + 1,
                'optim': optimizer.state_dict(),
            }
            save_ckpt(cpkt, ckpt_dir)
            best_ppl_val = val_ppl
            if verbose:
                print('Saved checkpoint!')

    if test_dataloader:
        if verbose:
            print('\nRunning against Test set:')
        best_ppl_test = reconstruction(
            encoder, decoder, test_dataloader, verbose=verbose)
        if tensorboard:
            writer.add_scalar('Perplexity/test', best_ppl_test, epoch)

    if tensorboard:
        writer.flush()
        writer.close()

    if verbose:
        print('Best\t| val ppl {:4.2f} | test ppl {:4.2f}'.format(
            best_ppl_val, best_ppl_test))

    return best_ppl_val, best_ppl_test
