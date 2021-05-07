import torch
from torch import nn
import torch.nn.functional as f

class PositionalEmbedding(nn.Module):
	def __init__(self, model_dim, seq_len):
		super().__init__()
		
		self.model_dim = model_dim
		pe = torch.zeros(seq_len,model_dim)

		for pos in range(seq_len):
			for i in range(0, model_dim, 2):
				pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
				pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))

		pe = pe.unsqueeze(0)
		self.register_buffer('pe',pe)

	def forward(self, x):
		return self.pe[:,:x.size(1)]


class FeedForward(nn.Module):
	def __init__(self, input_dim, inner_dim, dropout):
		super().__init__()

		self.linear_first = nn.Linear(input_dim,inner_dim)
		self.linear_second = nn.Linear(inner_dim, input_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.linear_second(self.dropout(f.relu(self.linear_first(x))))


class MultiHeadAttention(nn.Module):
	def __init__(self, heads, output, dropout):
		super().__init__()

		self.linear = nn.Linear(output, output * 3)

		self.heads = heads
		self.output = output
		
		self.out_per_head = output // heads 
		self.out = nn.Linear(output, output)

		self.dropout = nn.Dropout(dropout)

	def headsplit(self, t):
		return t.reshape(t.shape[0], -1, self.heads, self.out_per_head)

	def attention(self, q, k, v, mask, dropout):
		score = q.matmul(k.transpose(-2,-1))
		score /= math.sqrt(q.shape[-1])

		score = score if mask is None else score.masked_fill(mask == 0, -1e3)

		score = f.softmax(score, dim = -1)
		score = dropout(score) if dropout is not None else score

		return score.matmul(v)

	def forward(self, x, mask):
		
		y = x
		qkv = self.linear(x)
		
		q = qkv[:,:, :self.output]
		k = qkv[:,:, self.output:self.output*2]
		v = qkv[:,:, self.output*2:]

		q, k, v = [self.headsplit(t) for t in (q,k,v)]
		q, k, v = [t.transpose(1,2) for (q,k,v)]

		score = self.attention(q, k, v, mask, self.dropout)
		score = score.transpose(1,2).contiguous().view(score.shape[0], -1, self.output)

		return self.out(score)


class Encoder(nn.Module):

	def __init__(self, heads, transformer_size, feedforward_size, dropout):
		super().__init__()
		
		self.mha = MultiHeadAttention(heads,transformer_size,dropout)
		self.ff = FeedForward(transformer_size, feedforward_size, dropout)

		self.norm1 = nn.LayerNorm(transformer_size)
		self.norm2 = nn.LayerNorm(transformer_size)
			
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x, mask):
		
		x1 = self.norm1(x)
		x = x + self.dropout1(self.mha(x1, mask=mask))
		
		x2 = self.norm2(x)
		x = x + self.dropout2(self.ff(x2))

		return x 