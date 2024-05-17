import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from DeepSearch.model.models import *
from DeepSearch.training.distributed import *
from DeepSearch.training.scheduler import *
import logging
import os
import numpy as np
from torch.cuda.amp import GradScaler
import math
import statistics
from itertools import chain
from functools import partial
from typing import Set, Dict, Tuple, List
import torch.distributed as dist


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_search(spectra_latents: torch.Tensor,
                    pep_latents: torch.Tensor,
                    model: nn.Module,
                    device,
                    batch_data: Dict,
                    val_dataset,
                    mass,
                    database,
                    decoy,
                    pep_emds):

    peptides = batch_data['seq']
    batch_size = len(batch_data['seq'])

    candidates = retrieve_peps(mass, database, delta_mass=1)
    candidate_emds = retrive_pep_emds(candidates, pep_emds)
    with torch.no_grad():
        if len(candidate_emds) == 0:
            print(mass)
        all_pep_latents = torch.stack(candidate_emds).to(device)


        logits_per_spectra = (
            spectra_latents @ all_pep_latents.t()).detach().cpu()
        scores, ranking = logits_per_spectra.sort(descending=True)
        PSM_indices = ranking[:, 0].tolist()
        scores = scores[:, 0].tolist()

    n_decoy_hits = 0
    n_correct_hits = 0
    decoy_hit_score = []
    hit_score = []

    for i in range(batch_size):
        PSM_idx = PSM_indices[i]
        match_pep = candidates[PSM_idx]
        score = scores[i]
        if match_pep in decoy:
            n_decoy_hits += 1
            decoy_hit_score.append(score)
        else:
            hit_score.append(score)
        if match_pep == peptides[i]:
            n_correct_hits += 1

    return n_decoy_hits, n_correct_hits, decoy_hit_score, hit_score


def evaluate_pep(pred_pep: torch.Tensor, labels: torch.Tensor):
    """evaluate_pep return total_aa, predicted_total_aa, correct_total_aa 

    Args:
        pred_pep (torch.Tensor): [B, N=33]
        labels (torch.Tensor): [B, N=33]
    """
    b, n = labels.shape
    end_token = 24

    pep_len = ((labels != 0).sum(-1)) - 1

    # total amino_acid
    n_aa_total = pep_len.sum()

    # zero out the out-of-bound amino acid in predicted peptide
    filled_pred_pep = pred_pep.masked_fill(labels == 0, 33)
    # zero out end_token
    filled_pred_pep.masked_fill_(labels == end_token, 33)

    n_aa_correct = (filled_pred_pep == labels).sum()

    end_token_idx = torch.argmax(
        (pred_pep == end_token).to(dtype=torch.int), dim=-1)
    # if no end_token, set to N - 1
    end_token_idx[end_token_idx == 0] = n - 1
    n_aa_predicted = end_token_idx.sum()
    return n_aa_total, n_aa_predicted, n_aa_correct


def retrieve_peps(mass: float, db, delta_mass=2):
    candidates = []
    for delta in range(-delta_mass, delta_mass + 1, 1):
        rounded_mass = round(mass + delta, 0)
        if str(rounded_mass) not in db:
            continue
        candidates.extend(db[str(rounded_mass)])
    return candidates


def retrive_pep_emds(candidates, pep_emds):
    emds = []
    for candidate in candidates:
        emds.append(pep_emds[candidate])
    return emds


def train_one_epoch(model,
                    epoch_index,
                    train_loader,
                    train_sampler,
                    loss_fn,
                    optimizer,
                    accum_freq,
                    device,
                    scaler,
                    scheduler,
                    args,
                    tb_writer,
                    use_bias=True,
                    ):
    model.train()
    n_batch_per_epoch = len(train_loader)
    loss_m = {}
    train_sampler.set_epoch(epoch_index)
    # sample a large batch (e.g. 4096), distribute them onto devices use ranks
    # eg. 0-1023 on rank1, 1024-2047 on rank 2 ...
    # then use accum_freq
    for i, batch in enumerate(train_loader):

        step = n_batch_per_epoch * epoch_index + i
        scheduler(step)
        spectra = batch['spectrum']
        spectra_mask, pep_token =  batch['mask'], batch['t_peptide']
        pair_bias = None
        if use_bias:
            pair_bias = batch['pair_bias']

        world_batch_size = spectra.shape[0]
        # divide among device, drop last
        # batch_size is dividable by args.world_size * accum_freq
        batch_size = world_batch_size // args.world_size
        accum_batch_size = batch_size // accum_freq

        accum_spectra = spectra[batch_size *
                                args.rank: batch_size * (args.rank + 1), :, :].detach()
        accum_mask = spectra_mask[batch_size *
                                  args.rank: batch_size * (args.rank + 1), :].detach()
        accum_peps = pep_token[batch_size *
                               args.rank: batch_size * (args.rank + 1), :].detach()

        if use_bias:
            accum_pair_bias = pair_bias[batch_size *
                                        args.rank: batch_size * (args.rank + 1), :].detach()
        if accum_freq > 1:
            accum_spectra = torch.split(accum_spectra, accum_batch_size)
            accum_mask = torch.split(accum_mask, accum_batch_size)
            accum_peps = torch.split(accum_peps, accum_batch_size)
            if use_bias:
                accum_pair_bias = torch.split(
                    accum_pair_bias, accum_batch_size)
            accum_features = {}


        if accum_freq == 1:
            n_aa_total = 0
            n_aa_predicted = 0
            n_aa_correct = 0
            spectra = accum_spectra.to(device=device, non_blocking=True)
            spectra_mask = accum_mask.to(device=device, non_blocking=True)
            pep_token = accum_peps.to(device=device, non_blocking=True)
            if use_bias:
                pair_bias = accum_pair_bias.to(
                    device=device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_out = model(peptide=pep_token, spectra=spectra,
                                  spectra_mask=spectra_mask, 
                                  attn_bias=pair_bias)
                temperature = model_out.pop('temperature')
                loss = loss_fn(**model_out, temperature=temperature)
            scaler.scale(loss['loss']).backward()

            pred_pep = model_out['logits'].argmax(dim=-1)
            aa_total, aa_predicted, aa_correct = evaluate_pep(
                pred_pep, model_out['labels'])
            n_aa_total += aa_total
            n_aa_correct += aa_correct
            n_aa_predicted += aa_predicted

        else:
            for j in range(accum_freq):
                # gpu tensor are copied from accum_tensor
                spectra = accum_spectra[j].to(device=device, non_blocking=True)
                spectra_mask = accum_mask[j].to(
                    device=device, non_blocking=True)
                pep_token = accum_peps[j].to(device=device, non_blocking=True)

                if use_bias:
                    pair_bias = accum_pair_bias[j].to(
                        device=device, non_blocking=True)

                optimizer.zero_grad()
                # do a forward pass for each accumulative batch, without gradient, for caching latent
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        model_out = model(peptide=pep_token, spectra=spectra,
                                          spectra_mask=spectra_mask, attn_bias=pair_bias)
                        model_out.pop('temperature')
                        for key, val in model_out.items():
                            if key in accum_features:
                                #print(val.dtype)
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                if ((j + 1) != accum_freq):
                    continue

                # take gradients
                # re-do forward pass
                # calculate aa prediction accuracy
                n_aa_total = 0
                n_aa_predicted = 0
                n_aa_correct = 0
                optimizer.zero_grad()
                for k in range(accum_freq):
                    spectra = accum_spectra[k].to(
                        device=device, non_blocking=True)
                    spectra_mask = accum_mask[k].to(
                        device=device, non_blocking=True)
                    pep_token = accum_peps[k].to(
                        device=device, non_blocking=True)
                    if use_bias:
                        pair_bias = accum_pair_bias[k].to(
                            device=device, non_blocking=True)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                        model_out = model(peptide=pep_token, spectra=spectra,
                                          spectra_mask=spectra_mask, attn_bias=pair_bias)
                        temperature = model_out.pop('temperature')
                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(
                                accumulated[:k] + [model_out[key]] + accumulated[k + 1:]) if model_out[key] is not None else None
                        # with torch.cuda.amp.autocast(enabled=False):
                        loss = loss_fn(**inputs, temperature=temperature)
                        del inputs

                    # make prediction
                    pred_pep = accum_features['logits'][k].argmax(dim=-1)
                    aa_total, aa_predicted, aa_correct = evaluate_pep(
                        pred_pep, accum_features['labels'][k])
                    n_aa_total += aa_total
                    n_aa_correct += aa_correct
                    n_aa_predicted += aa_predicted
                    scaler.scale(loss['loss'] / accum_freq).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            unwrap_model(model).temperature.clamp_(0, math.log(100))
        if accum_freq > 1:
            del accum_features

        aa_precision = (n_aa_correct / (n_aa_predicted + 1e-7)).item()
        aa_recall = (n_aa_correct / (n_aa_total + 1e-7)).item()

        # logging
        if is_master(args) and (i % 20 == 0 or (i + 1) == n_batch_per_epoch):
            for k, v in loss.items():
                if k not in loss_m:
                    loss_m[k] = AverageMeter()
                loss_m[k].update(v.item(), world_batch_size)

            log_data = {
                "aa_precision": aa_precision,
                "aa_recall": aa_recall
            }
            log_data.update({k: v.val for k, v in loss_m.items()})

            for k, v in log_data.items():
                k = "train/" + k
                if tb_writer is not None:
                    tb_writer.add_scalar(k, v, step)

            logging.info(f"     batch {i + 1}: loss: {loss_m['loss'].val}; contrastive: {loss_m['contrastive_loss'].val};"
                         f"caption: {loss_m['caption_loss'].val}; aa_precision: {aa_precision}; aa_recal: {aa_recall}"
                         )


def _collate_sharded_tensor(sharded_tensor, shape):
    out_tensor = torch.empty(shape, device=torch.device("cpu"))
    dist.all_gather_into_tensor(out_tensor, sharded_tensor.cpu())
    # sharded_tensor.cpu(process_group=pg).gather(0, out_tensor)
    # print(out_tensor.shape)
    return out_tensor


def cal_database_emd_dist(model, tokenizer, database_withdecoy: OrderedDict[Tuple[str, List]], args, device):
    emd_database = {}

    with torch.no_grad():
        for _, peptides in database_withdecoy.items():
            n_peptides = len(peptides)
            # print(peptides)
            chunk_size = n_peptides // args.world_size
            s = args.rank * chunk_size
            e = s + chunk_size
            if e > n_peptides:
                e = n_peptides
            chunk_peptides = peptides[s: e]
            if len(chunk_peptides) == 0:
                continue
            chunk_tokens = torch.stack(
                list(map(tokenizer.encode, chunk_peptides)), dim=0)
            chunk_tokens = chunk_tokens.to(device)
            chunk_emd = model(None, None, chunk_tokens, return_pep_emd=True)

            # gather chunk_emd from device to cpu
            if is_master(args):
                shape = [n_peptides, chunk_emd.shape[1]]
                all_emd = _collate_sharded_tensor(chunk_emd, shape)
                assert all_emd.shape[0] == n_peptides
                for i, pep in enumerate(peptides):
                    emd_database[pep] = all_emd[i, :].detach()
    return emd_database


def cal_database_emd(model, tokenizer, database_withdecoy: OrderedDict[Tuple[str, List]], args, device):
    if not is_master(args):
        return None
    all_peptides = list(database_withdecoy.values())
    all_peptides = [item for sublist in all_peptides for item in sublist]
    emd_database = OrderedDict()
    B = 2048
    with torch.no_grad():
        n_peptides = len(all_peptides)
        n_chunk = n_peptides // B + 1
        for i in range(n_chunk):
            s = i*B
            e = s + B
            if e > n_peptides:
                e = n_peptides
            chunk_peptides = all_peptides[s: e]
            if len(chunk_peptides) == 0:
                continue
            chunk_tokens = torch.stack(
                list(map(tokenizer.encode, chunk_peptides)), dim=0)
            chunk_tokens = chunk_tokens.to(device)
            chunk_emd = model(None, None, chunk_tokens, return_pep_emd=True)
            for j, pep in enumerate(chunk_peptides):
                emd_database[pep] = chunk_emd[j, :].cpu().detach()
    return emd_database


def evaluate(model, 
             val_set,
             val_loader, 
             val_sampler, 
             epoch, 
             device, 
             args, 
             tb_writer, 
             use_bias=True, ):
    if not is_master(args):
        return None

    model.eval()
    cumulative_con_loss = 0.0
    cumulative_cap_loss = 0.0
    n_samples = 0
    n_aa_total = 0
    n_aa_predicted = 0
    n_aa_correct = 0

    n_sample_per_val = len(val_set)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            
            spectra = batch['spectrum']
            spectra_mask, pep_token =  batch['mask'], batch['t_peptide']
            batch_size = spectra.shape[0]
            spectra = spectra.to(device=device, non_blocking=True)
            spectra_mask = spectra_mask.to(device=device, non_blocking=True)
            pep_token = pep_token.to(device=device, non_blocking=True)
            pair_bias = None

            if use_bias:
                pair_bias = batch['pair_bias'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_out = model(peptide=pep_token, spectra=spectra,
                                  spectra_mask=spectra_mask, 
                                  attn_bias=pair_bias)
                temperature = model_out.pop('temperature')

                logits_per_spectra = temperature * \
                    model_out['spectra_latents'] @ model_out['pep_latents'].t()
                logits_per_pep = logits_per_spectra.t()

                batch_size = spectra.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                con_loss = (
                    F.cross_entropy(logits_per_spectra, labels) +
                    F.cross_entropy(logits_per_pep, labels)
                ) / 2

                cap_loss = F.cross_entropy(model_out['logits'].permute(
                    0, 2, 1), model_out['labels'], ignore_index=0)

                cumulative_con_loss += con_loss * batch_size
                cumulative_cap_loss += cap_loss * batch_size
                n_samples += batch_size

                # evaluate de novo result
                pred_pep = model_out['logits'].argmax(dim=-1)
                aa_total, aa_predicted, aa_correct = evaluate_pep(
                    pred_pep, model_out['labels'])
                n_aa_total += aa_total
                n_aa_correct += aa_correct
                n_aa_predicted += aa_predicted

                if is_master(args) and (i % 400) == 0:
                    logging.info(
                        f'Validation Epoch {epoch}: {epoch} [{n_samples} / {n_sample_per_val}] \t'
                        f'contrastive: {cumulative_con_loss / n_samples}; caption: {cumulative_cap_loss / n_samples};'
                        f'aa_precision: {(n_aa_correct / (n_aa_predicted + 1e-7)).item()};'
                        f'aa_recal: {(n_aa_correct / (n_aa_total + 1e-7)).item()}.'
                    )

        cumulative_cap_loss /= n_samples
        cumulative_con_loss /= n_samples
        aa_precision = (n_aa_correct / (n_aa_predicted + 1e-7)).item()
        aa_recall = (n_aa_correct / (n_aa_total + 1e-7)).item()


        if is_master(args):
            logging.info(
                f'Validation Epoch {epoch}: \t'
                f'contrastive: {cumulative_con_loss}; caption: {cumulative_cap_loss}; \t'
                f'aa_precision: {aa_precision};'
                f'aa_recall: {aa_recall}; \n'
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    f"val/contrastive_loss", cumulative_con_loss, epoch)
                tb_writer.add_scalar(
                    f"val/caption_loss", cumulative_cap_loss, epoch)
                tb_writer.add_scalar(
                    f"val/aa_precision", aa_precision, epoch)
                tb_writer.add_scalar(
                    f"val/aa_recall", aa_recall, epoch)

    return
