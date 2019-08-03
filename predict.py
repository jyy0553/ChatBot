import sys
import numpy as np 
import configure
import tensorflow as tf
# from data_helpers import loadDataset, getBatches, sentence2encoder
from pre_dataset import loadDataset,getBatches,sentence2encode
from model import Seq2SeqModel
# FLAGS = configure.flages.FLAGS
# FLAGS.flag_values_dict()
FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()

data_path = "data/dataset-cornell-length10-filter1-vocabSize40000.pkl"
word2id,id2word,trainingSamples = loadDataset(data_path)

def predict_ids_to_seq(predict_ids, id2word,beam_size):
	for single_predict in predict_ids:
		# print(single_predict.shape)
		# exit()
		# for i in range(beam_size):
		for i in range(1):
			predict_list = np.ndarray.tolist(single_predict[:,:,i])
			predict_seq = [id2word[idx] for idx in predict_list[0]]
			print(" ".join(predict_seq))

with tf.Session() as sess:
	model = Seq2SeqModel(FLAGS.rnn_size,FLAGS.num_layers,FLAGS.embedding_size,FLAGS.learning_rate,
						word2id,mode = 'decode',use_attention = True,beam_search = False, beam_size = 5,max_gradient_norm = 5.0)
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Reloading model parameters...')
		model.saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		raise ValueError("No such file:{}".format(FLAGS.model_dir))
	sys.stdout.write(">")
	sys.stdout.flush()
	sentence = sys.stdin.readline()
	# print(sentence)
	while sentence:
		batch = sentence2encode(sentence,word2id)
		predicted_ids = model.infer(sess,batch)
		predict_ids_to_seq(predicted_ids,id2word,5)
		print(">","")
		sys.stdout.flush()
		sentence = sys.stdin.readline()