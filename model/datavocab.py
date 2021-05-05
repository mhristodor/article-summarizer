import random
import torch
import numpy as np
import re
from utils import prepareTraining

class Data(object):
	def __init__(self, file, maxlen):
		self.file = file
		self.maximum = maxlen

	def load(self):
		
		f = open(self.file,"r",encoding = "utf-8")
		pages = [[]]
		
		for line in f.readlines():
			tokens = line.strip().split()
			if tokens:
				pages[-1].append(tokens)
			else:
				pages.append([])
		
		pages = [_ for _ in pages if _]
		random.shuffle(pages)

		data = []
		for index, page in enumerate(pages):
			print(page)
			print(prepareTraining(pages,index,self.maximum))
			exit()
			data.extend(prepareTraining(pages,index,self.maximum))

		return data


class Vocab(object):
	def __init__(self, file, minim):
		
		PAD, UNK, CLS, SEP, MASK, NUM, NOT_ROMANIAN = '<-PAD->', '<-UNK->', '<-CLS->', '<-SEP->', '<-MASK->', '<-NUM->', '<-NOT_ROMANIAN->'
		number_regex = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
		tokens = [PAD, UNK, NUM, NOT_ROMANIAN, CLS, SEP, MASK]

		f = open(file,"r",encoding = "utf-8")

		for line in f.readlines():
			token, nr = line.strip().split()
			nr = int(nr)

			if number_regex.match(token):
				continue

			if self.notRomanian(token) and nr >= 2.5 * minim:
				tokens.append(token)
			elif not self.notRomanian(token) and nr >= minim:
				tokens.append(token)

		self.id = dict(zip(tokens, range(len(tokens))))
		self.toekns = tokens

		self.not_rom_id = self.id[NOT_ROMANIAN]
		self.padding_id = self.id[PAD]
		
		self.unk_id = self.id[UNK]
		self.num_id = self.id[NUM]

	def notRomanian(self,text):
		for c in text:
			nr = ord(c)

			if nr > 126 and nr not in [258,259,194,226,206,238,218,219,350,351,538,539,354,355]:
				return True

		return False


data = Data(r"F:\corpus\parsed_corpus.txt",128)
data.load()

