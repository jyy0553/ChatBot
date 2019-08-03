import os
import nltk
import numpy as np 
import pickle
import random

padToken, goToken, eosToken, unknownToken = 0,1,2,3

class Batch:
	def __init__(self):
		self.encoder_inputs = []
		self.encoder_inputs_length = []
		self.decoder_targets = []
		self.decoder_targets_length = []

def loadDataset(filename):
	dataset_path = os.path.join(filename)
	print("loading dataset from{}".format(dataset_path))
	with open(dataset_path,'rb') as handle:
		data = pickle.load(handle)
		word2id = data['word2id']
		id2word = data['id2word']
		trainingSamples = data['trainingSamples']
	print(type(trainingSamples))
	return word2id,id2word,trainingSamples

# data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
# word2id,id2word,trainingSamples = loadDataset(data_path)
# for i in trainingSamples:
# 	print(i[0])
# 	break

def createBatch(samples):
	batch = Batch()
	batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
	batch.decoder_targets_length = [len(sample[1]) for sample in samples]

	max_source_length = max(batch.encoder_inputs_length)
	max_target_length = max(batch.decoder_targets_length)

	for sample in samples:
		source = list(reversed(sample[0]))
		pad = [padToken] * (max_source_length - len(source))
		batch.encoder_inputs.append(pad+source)

		target = sample[1]
		pad = [padToken] * (max_target_length - len(target))
		batch.decoder_targets.append(target + pad)

	return batch

# def getBatches(data, batch_size):
# 	random.shuffle(data)
# 	batches = []
# 	data_len = len(data)
# 	def genNextSamples():
# 		for i in range(0,data_len, batch_size):
# 			yield data[i:min(i+batch_size, data_len)]

# 	for samples in genNextSamples():
# 		batch = createBatch(samples)
# 		batches.append(batch)

# 	return batches

def getBatches(data, batch_size):
	random.shuffle(data)
	batches = []
	data_len = len(data)
	n_batches = int(data_len*1.0/batch_size)
	for i in range(0,n_batches):
		batch = data[i*batch_size:(i+1)*batch_size]
		batch = createBatch(batch)
		yield batch



def sentence2encode(sentence, word2id):
	if sentence == '':
		return None

	tokens = nltk.word_tokenize(sentence)
	if len(tokens) > 20:
		return None

	wordIds = []
	for token in tokens:
		wordIds.append(word2id.get(token, unknownToken))

	batch = createBatch([[wordIds,[]]])
	return batch