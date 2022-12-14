# Copied from https://github.com/wxjiao/AGHMN
__author__ = "Wenxiang Jiao"

"""
Utils

"""
import json
import pickle
import torch
import os
import math
import random
import numpy as np
import Const
import time



# Timer
def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def ToTensor(list, is_len=False):
	np_ts = np.array(list)
	tensor = torch.from_numpy(np_ts).long()

	if is_len:
		mat1 = np.equal(np_ts, Const.PAD)
		mat2 = np.equal(mat1, False)
		lens = np.sum(mat2, axis=1)
		return tensor, lens

	return tensor

# model saver
def model_saver(model, path, module, dataset):
	if not os.path.isdir(path):
		os.makedirs(path)
	model_path = '{}/{}_{}.pt'.format(path, module, dataset)
	torch.save(model, model_path)

# model loader
def model_loader(path, module, dataset):
	model_path = '{}/{}_{}.pt'.format(path, module, dataset)
	model = torch.load(model_path, map_location='cpu')
	return model


def saveToJson(path, object):
	t = json.dumps(object, indent=4)
	f = open(path, 'w')
	f.write(t)
	f.close()

	return 1


def saveToPickle(path, object):
	file = open(path, 'wb')
	pickle.dump(object, file)
	file.close()

	return 1


def loadFrPickle(path):
	file = open(path, 'rb')
	obj = pickle.load(file)
	file.close()

	return obj


def load_bin_vec(filename, vocab):
	"""
	Loads 300x1 word vecs from Google (Mikolov) word2vec
	dtype: word2vec float32, glove float64;
	Word2vec's input is encoded in UTF-8, but output is encoded in ISO-8859-1
	"""
	print('Initilaize with Word2vec 300d word vectors!')
	word_vecs = {}
	with open(filename, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split()[0:2])
		binary_len = np.dtype('float32').itemsize * layer1_size
		num_tobe_assigned = 0
		for line in range(vocab_size):
			word = []
			while True:
				ch = f.read(1).decode('iso-8859-1')
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)
			if word in vocab:
				vector = np.fromstring(f.read(binary_len), dtype='float32')
				word_vecs[word] = vector / np.sqrt(sum(vector**2))
				num_tobe_assigned += 1
			else:
				f.read(binary_len)
		print("Found words {} in {}".format(vocab_size, filename))
		match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
		print("Matched words {}, matching rate {} %".format(num_tobe_assigned, match_rate))
	return word_vecs


def load_txt_glove(filename, vocab):
	"""
	Loads 300x1 word vecs from Glove
	dtype: glove float64;
	"""
	print('Initilaize with Glove 300d word vectors!')
	word_vecs = {}
	vector_size = 300
	with open(filename, "r") as f:
		vocab_size = 0
		num_tobe_assigned = 0
		for line in f:
			vocab_size += 1
			splitline = line.split()
			word = " ".join(splitline[0:len(splitline) - vector_size])
			if word in vocab:
				vector = np.array([float(val) for val in splitline[-vector_size:]])
				word_vecs[word] = vector / np.sqrt(sum(vector**2))
				num_tobe_assigned += 1

		print("Found words {} in {}".format(vocab_size, filename))
		match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
		print("Matched words {}, matching rate {} %".format(num_tobe_assigned, match_rate))
	return word_vecs


def load_pretrain(d_word_vec, diadict, type='word2vec'):
	""" initialize nn.Embedding with pretrained """
	if type == 'word2vec':
		filename = 'Data/GoogleNews-vectors-negative300.bin'
		word2vec = load_bin_vec(filename, diadict.word2index)
	elif type == 'glove':
		filename = 'Data/glove.840B.300d.txt'
		word2vec = load_txt_glove(filename, diadict.word2index)

	# initialize a numpy tensor
	embedding = np.random.uniform(-0.01, 0.01, (diadict.n_words, d_word_vec))
	for w, v in word2vec.items():
		embedding[diadict.word2index[w]] = v

	# zero padding
	embedding[Const.PAD] = np.zeros(d_word_vec)

	return embedding


def load_char_vec(filename, vocab):
	"""
	Loads 300x1 char vecs from glove.840B.300d-char.txt
	dtype: glove float64;
	UTF-8, but output is encoded in ISO-8859-1
	"""
	char_vecs = {}
	with open(filename, "r") as f:
		vocab_size = 94
		layer_size = 300
		num_tobe_assigned = 0
		for line in f:
			splitline = line.split()
			char = splitline[0]
			if char in vocab:
				vector = np.array([float(v) for v in splitline[1:]])
				assert len(vector) == layer_size
				char_vecs[char] = vector / np.sqrt(sum(vector**2))
				num_tobe_assigned += 1
		print("Found chars {} in {}".format(vocab_size, filename))
		match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
		print("Matched chars {}, matching rate {} %".format(num_tobe_assigned, match_rate))
	return char_vecs


def load_charvecs(d_char_vec, n_chars, char2idx):
	""" initialize nn.Embedding with pretrained """
	filename = 'glove.840B.300d-char.txt'
	char2vec = load_char_vec(filename, char2idx)

	# initialize a numpy tensor
	embedding = np.random.uniform(-0.01, 0.01, (n_chars, d_char_vec))
	for c, v in char2vec.items():
		embedding[char2idx[c]] = v

	# zero padding
	embedding[Const.cPAD] = np.zeros(d_char_vec)

	return embedding



def shuffle_lists(featllist, labellist=None, thirdparty=None):

	if labellist == None:
		random.shuffle(featllist)
		return featllist
	elif labellist != None and thirdparty == None:
		combined = list(zip(featllist, labellist))
		random.shuffle(combined)
		featllist, labellist = zip(*combined)
		return featllist, labellist
	else:
		combined = list(zip(featllist, labellist, thirdparty))
		random.shuffle(combined)
		featllist, labellist, thirdparty = zip(*combined)
		return featllist, labellist, thirdparty


def param_clip(model, optimizer, batch_size, max_norm=10):
	# gradient clipping
	shrink_factor = 1
	total_norm = 0

	for p in model.parameters():
		if p.requires_grad:
			p.grad.data.div_(batch_size)
			total_norm += p.grad.data.norm() ** 2
	total_norm = np.sqrt(total_norm)

	if total_norm > max_norm:
		# print("Total norm of grads {}".format(total_norm))
		shrink_factor = max_norm / total_norm
	current_lr = optimizer.param_groups[0]['lr']

	return current_lr, shrink_factor