from datavocab import Data, Vocab
from dataset import MyDataset
from bert import Bert
from utils import get_batch
import torch.nn as nn
import gc

import numpy as np
import torch

batch_size = 1
seq_len = 128
embed = 128
workers = 0
ff = 128 * 4
heads = 8
layers = 12
dropout = 0.1
iterrations = 50000


optimizer = {'lr':2e-3, 'weight_decay':1e-4, 'betas':(.9,.999)}
loader_settings = {'num_workers':workers, 'shuffle':False,  'drop_last':True, 'pin_memory':False, 'batch_size':batch_size}


data = Data(r"F:\corpus\parsed_corpus.txt",seq_len)
content = data.load()
vocab = Vocab(r"F:\corpus\vocab.txt",1)
dataset = MyDataset(content,vocab,seq_len)

del content
del vocab
gc.collect()

loader = torch.utils.data.DataLoader(dataset, **loader_settings)

model = Bert(layers,heads,embed,ff,len(dataset.vocab.id),seq_len,dropout)

model.cuda()

adam = torch.optim.Adam(model.parameters(), **optimizer)
loss_model = nn.CrossEntropyLoss(ignore_index = dataset.vocab.padding_id)

pulse = 5

model.train()

batch_iter = iter(loader)

for it in range(iterrations):

	batch, batch_iter = get_batch(loader,batch_iter)
	
	mask_in = batch['input']
	mask_target = batch['target']

	mask_in = mask_in.cuda(non_blocking = True)
	mask_target = mask_target.cuda(non_blocking = True)
	output = model(mask_in)

	out = output.view(-1,output.shape[-1])
	target_o = mask_target.view(-1,1).squeeze()

	loss = loss_model(out,target_o)
	
	loss.backward()
	adam.step()

	if it % pulse == 0:
		print("Iteration: ", it, " Loss: ", np.round(loss.item(),2)," DeltaW: ", round(model.embeddings.weight.grad.abs().sum().item(),3))

	adam.zero_grad()


np.savetxt("weight.tsv",np.round(model.embeddings.weight.detach().cpu().numpy()[0:N],2), delimiter='\t', fmt='%1.2f')