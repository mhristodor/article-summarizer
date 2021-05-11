from torch.utils.data import Dataset
import random
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, sent, vocab, seq_len):

        tmp = [] 
        fullsent = []

        for s in sent:
            tmp.extend(s[0])
            tmp.extend(s[1])
            fullsent.append(tmp)
            tmp = []


        self.sent = fullsent

        self.vocab = vocab
        self.seq_len = seq_len
 
    def __getitem__(self, index, randomness = 0.15):

        sent = self.sent_id(index)

        for i in range(self.seq_len - len(sent)):
            sent.append(self.vocab.padding_id)

        masked = [[self.vocab.mask_id,word] if random.random() < randomness else [word, self.vocab.padding_id] for word in sent]
        
        for mask in masked:
            if mask[0] == self.vocab.mask_id and random.random() < 0.1:
                mask[0] = mask[1]
            elif random.random() < 0.1:
                entry_list = list(self.vocab.id.items())
                random_entry = random.choice(entry_list)
                mask[0] = random_entry[1]


        return {'input': torch.Tensor([word[0] for word in masked]).long(),
                'target': torch.Tensor([word[1] for word in masked]).long()}


    def __len__(self):
        return len(self.sent)

    def sent_id(self, index):
       
        s = self.sent[index]
        s = [self.vocab.id[w] if w in self.vocab.id else self.vocab.not_rom_id for w in s] 
        
        return s