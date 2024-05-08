import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.distributed.nn 


def gather_features(spectra_latents, pep_latents, decoy_latents=None):
    all_spectra_latents = torch.cat(
        torch.distributed.nn.all_gather(spectra_latents), dim=0)
    all_pep_latents = torch.cat(
        torch.distributed.nn.all_gather(pep_latents), dim=0)
    if decoy_latents is not None:
        all_decoy_latents = torch.cat(
            torch.distributed.nn.all_gather(decoy_latents), dim=0)

        return all_spectra_latents, all_pep_latents, all_decoy_latents
    return all_spectra_latents, all_pep_latents


def log_softmax(x, eps = 1e-7):
    return x - (x.exp().sum(-1) + eps).log().unsqueeze(-1)


def cross_entropy(preds, targets, ignore_index = -100):
    x = F.nll_loss(log_softmax(preds), targets, ignore_index=ignore_index)
    return x
    
    

class CocaLoss(nn.Module):
    def __init__(self, 
                 caption_loss_weight = 2.0, 
                 contrastive_loss_weight=1.0, 
                 pad_id=0, 
                 unique_label=True,
                 rank=0,
                 world_size=1):
        super().__init__()
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.pad_id = pad_id
        self.unique_label = unique_label
        self.rank = rank
        self.world_size = world_size

        self.prev_batch_size = 0
        self.labels = {}
    
    
    def get_ground_truth(self, device, batch_size):
        if self.prev_batch_size != batch_size or (device not in self.labels):
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.world_size > 1:
                 labels = labels + batch_size * self.rank
            self.labels[device] = labels
            self.prev_num_logits = batch_size
        else:
            labels = self.labels[device]
        
        return labels
    

    def get_logits(self, spectra_latents, pep_latents, temperature, decoy_latents=None):
        if decoy_latents is None:
            if self.world_size > 1:
                all_spectra_latents, all_pep_latents = gather_features(spectra_latents, pep_latents)
                logits_per_spectra = temperature * spectra_latents @ all_pep_latents.T 
                logits_per_pep = temperature * pep_latents @ all_spectra_latents.T
            else: 
                logits_per_spectra = temperature * spectra_latents @ pep_latents.T 
                logits_per_pep = logits_per_spectra.t()
        else:
            if self.world_size > 1:
                all_spectra_latents, all_pep_latents, all_decoy_latents = gather_features(
                    spectra_latents, pep_latents, decoy_latents)
                all_pep_latents_ = torch.cat([all_pep_latents, all_decoy_latents])

                logits_per_spectra = temperature * spectra_latents @ all_pep_latents_.T
                logits_per_pep = temperature * pep_latents @ all_spectra_latents.T
        
            else:
                pep_latents_ = torch.cat([pep_latents, decoy_latents])
                logits_per_spectra = temperature * spectra_latents @ pep_latents_.T
                logits_per_pep = temperature * pep_latents @ spectra_latents.T
                
        return logits_per_spectra, logits_per_pep


    def forward(self, spectra_latents, pep_latents, logits, labels, temperature, decoy_latents=None):
        device = spectra_latents.device
        
        logits_per_spectra, logits_per_pep = self.get_logits(
            spectra_latents, pep_latents, temperature, decoy_latents)

        batch = logits_per_spectra.shape[0]
        contrastive_labels = self.get_ground_truth(device, batch)

        contrastive_loss = (F.cross_entropy(logits_per_spectra, contrastive_labels) +
                            F.cross_entropy(logits_per_pep, contrastive_labels)) / 2
        
        logits = rearrange(logits, 'b n c -> b c n')
        caption_loss = F.cross_entropy(
            logits, labels, ignore_index=self.pad_id)
        
        loss = self.contrastive_loss_weight * contrastive_loss + \
            self.caption_loss_weight * caption_loss
        
        return {"contrastive_loss": contrastive_loss,
                "caption_loss": caption_loss,
                "loss": loss}
