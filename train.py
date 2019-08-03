import tensorflow as tf 
from pre_dataset import loadDataset,getBatches,sentence2encode
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os
import configure

FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()

data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id,id2word,trainingSamples = loadDataset(data_path)
with tf.Graph().as_default():
	with tf.device("/cpu:0"):
		session_conf = tf.ConfigProto()
		session_conf.allow_soft_placement = FLAGS.allow_soft_placement
		session_conf.log_device_placement = FLAGS.log_device_placement
		session_conf.gpu_options.allow_growth = True
	sess = tf.Session(config = session_conf)
	with sess.as_default():
		model = Seq2SeqModel(FLAGS.rnn_size,FLAGS.num_layers,FLAGS.embedding_size,FLAGS.learning_rate,
							word2id,
							mode = 'train',
							use_attention = True,
							beam_search = False,
							beam_size = 5,
							max_gradient_norm = 5.0)

		ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print("Reloading model parameters...")
			model.restore(sess,ckpt.model_checkpoint_path)
		else:
			print("created new model parameters...")
			sess.run(tf.global_variables_initializer())
		# sess.run(tf.global_variables_initializer())
		current_step = 0
		# summary_writer = tf.summary.FileWriter(FLAGS.model_dir,graph = sess.graph)
		for e in range(FLAGS.num_epochs):
			print("------Epoch {}/{} -----".format(e+1,FLAGS.num_epochs))
			batches = getBatches(trainingSamples,FLAGS.batch_size)
			for nextBatch in tqdm(batches,desc = 'Training'):
				# print(len(nextBatch.encoder_inputs))
				# print(nextBatch.encoder_inputs)
				# print("**************")
				# print(nextBatch.decoder_targets)
				# exit()
				# loss,summary = model.train(sess,nextBatch)
				loss = model.train(sess,nextBatch)
				current_step += 1
				if current_step % FLAGS.steps_per_checkpoint == 0:
					perplexity = math.exp(float(loss)) if loss< 300 else float('inf')
					tqdm.write("----- Step %d -- Loss %.2f -- perplexity %.2f" %(current_step, loss, perplexity))
					# summary_writer.add_summary(summary,current_step)
					checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
					model.saver.save(sess,checkpoint_path,global_step = current_step)