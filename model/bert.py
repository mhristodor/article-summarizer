import torch
from torch import nn

from transformer import Encoder, PositionalEmbedding


class Bert(nn.Module):
	def __init__(self, encoder_count, heads, embed_size, feedforward_size, embeddings, seq_len, dropout):
		super().__init__()

		self.embeddings = nn.Embedding(embeddings,embed_size)
		self.pe = PositionalEmbedding(embed_size, seq_len)

		encoders = [] 
		for _ in range(encoder_count):
			encoders += [Encoder(heads, embed_size, feedforward_size, dropout)]
		
		self.encoders = nn.ModuleList(encoders)

		self.norm = nn.LayerNorm(embed_size)
		self.linear = nn.Linear(embed_size, embeddings, bias = False)

	def forward(self, x):
		
		x = self.embeddings(x)
		x = x + self.pe(x)

		for encoder in self.encoders:
			x = encoder(x)

		x = self.norm(x)
		x = self.linear(x)

		return x 