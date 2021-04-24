import random
import torch
import numpy as np
import re
from utils import prepareTraining

class Data(object):
	def __init__(self, vocab, file, batch_size, maxlen):
		self.file = file
		self.f = open(self.file, encoding="utf-8")
		self.batch_size = batch_size
		self.vocab = vocab
		self.maximum = maxlen
		self.iterations = 0