from typing import List
import torch 
from DeepSearch.utils.peptide import *
from line_profiler import profile
class Tokenizer():
    def __init__(self, pep_len=32) -> None:
        self.vocab = AA_INDEX
        self.pep_len = pep_len
        self.tokens = list(self.vocab.keys())
        self.pad_token = '<pad>'
        self.pad_token_id = self.vocab[self.pad_token] 
        self.start_token = '<s>'
        self.start_token_id = self.vocab[self.start_token]
        self.end_token = '<e>'
        self.end_token_id = self.vocab[self.end_token]
    

    
    def tokenize(self, pep: str) -> List[str]:
        return [self.vocab[aa] for aa in pep]
    
    def token_to_id(self, token):
        return self.vocab[token]
    
    def tokens_to_id(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]
    
    def id_to_token(self, index: int) -> str:
        return self.tokens[index]
    
    def ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.id_to_token(id_) for id_ in indices]
    
    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        pep_len = len(token_ids)
        tokens = [self.pad_token_id] * (self.pep_len + 2)
        tokens[0] = self.start_token_id
        for i in range(pep_len):
            tokens[i+1] = token_ids[i]
        tokens[pep_len + 1] = self.end_token_id
        return tokens 

    def decode(self):
        pass 
    
    def encode(self, pep: str) -> torch.Tensor:
        tokens = self.tokenize(pep)
        tokens = self.add_special_tokens(tokens)
        #tokens_ids = self.tokens_to_id(tokens)
        return torch.tensor(tokens, dtype=torch.long)