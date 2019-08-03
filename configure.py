# import tensorflow as tf 
from tensorflow import flags
flags.DEFINE_integer('rnn_size',1024,'Number of hidden units in each layer')
flags.DEFINE_integer('num_layers',2,'Number of layer in each encoder an decoder')
flags.DEFINE_integer('embedding_size',1024,'Embedding dimensions of encoder and decoder inputs')


flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
flags.DEFINE_integer('batch_size',128,'Batch size')
flags.DEFINE_integer('num_epochs',30,'Maximum # of training epochs')
flags.DEFINE_integer('steps_per_checkpoint',100,'Save model checkpoint every this iteration')
flags.DEFINE_string('model_dir','model/','Path to save model checkpoints')
flags.DEFINE_string('model_name','chatbot.ckpt','File name used for model checkpoints')

flags.DEFINE_boolean("allow_soft_placement",True,"Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement",False,"Log placement of ops on devices")