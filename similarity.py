import sys
import argparse
import csv
import math
import numpy as np
import tensorflow as tf

max_userId = 0
max_movieId = 0 
filenames = []
num_epochs = 50
batch_size = 128
min_after_dequeue = batch_size 
capacity = batch_size * 3 + min_after_dequeue
hidden_width = 20
embedding_width = 100

def linear_block(x, in_width, out_width, name):
	w = tf.Variable(tf.random_normal([in_width, out_width], stddev=1e-2))
	b = tf.Variable(tf.zeros([out_width,]))
	tf.summary.histogram(name + "_w", w)
	#tf.summary.histogram(name + "_b", b)
	#return w, b, tf.matmul(x, w) + b
	return w, b, tf.matmul(x, w)

def full_block(x, in_width, out_width, activation=tf.nn.sigmoid):
	w, b, a = linear_block(x, in_width, out_width)
	return w, b, activation(a)

def res_block(x, width, activation, num_blocks):
	w = []
	b = []
	for i in range(num_blocks):
		w1, b1, a = full_block(x, width, width, activation)
		w2, b2, a = full_block(a, width, width, activation)
		x = x + a
		w.append(w1)
		w.append(w2)
		b.append(b1)
		b.append(b2)
	print((w, b))
	return w, b, x
	
def file_reader_graph_join(filenames, num_epochs):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=num_epochs)
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	example_list = [tf.decode_csv(value, record_defaults=[[0], [0], [0.0], [1]]) for _ in range(4)]
	print(example_list)
	example_list = [[tf.stack([e[0], e[1]]), e[2]] for e in example_list]
	print(example_list)
	return example_list

def file_reader_graph(filenames, num_epochs):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=num_epochs)
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	
	uid, mid, rating, time = tf.decode_csv(value, record_defaults=[[0], [0], [0.0], [1]])
	example = tf.stack([uid, mid])
	#rating = tf.stack([rating])
	print((example, rating))
	return example, rating

def batch_graph_join(example_list, batch_size, capacity, min_after_dequeue, shuffle=False):
	if shuffle:
		example_batch, rating_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, \
			capacity=capacity, min_after_dequeue=min_after_dequeue, enqueue_many=False)
	else:
		example_batch, rating_batch = tf.train.batch_join(example_list, batch_size=batch_size, enqueue_many=False)
	print((example_batch, rating_batch))
	return example_batch, rating_batch

def batch_graph(example, rating, batch_size, capacity, min_after_dequeue, shuffle=False, num_threads=1):
	if shuffle:
		example_batch, rating_batch = tf.train.shuffle_batch([example, rating], batch_size=batch_size, \
			num_threads=num_threads, capacity=capacity, min_after_dequeue=min_after_dequeue, enqueue_many=False)
	else:
		example_batch, rating_batch = tf.train.batch([example, rating], batch_size=batch_size, num_threads=num_threads, enqueue_many=False)
	print(example_batch)
	print(rating_batch)
	return example_batch, rating_batch

def embedding_variable(maxId, embedding_width, name=None):
	embedding = tf.Variable(tf.random_normal([maxId + 1, embedding_width], mean=0.0, stddev=1.0/math.sqrt(embedding_width)))
	return embedding

def net():
	L = []
	alpha = []
	with tf.name_scope("embedding"):
		#with tf.device("/cpu:0"):
		user_embedding = embedding_variable(max_userId, embedding_width, name='user_embedding')
		movie_embedding = embedding_variable(max_movieId, embedding_width, name='movie_embedding')
	
	with tf.name_scope("file_reader"):
		with tf.device("/cpu:0"):
			#example, rating = file_reader_graph(filenames, num_epochs)
			example_list = file_reader_graph_join(filenames, num_epochs)
	
	with tf.name_scope("batch"):
		with tf.device("/cpu:0"):
			#example_batch, rating_batch = batch_graph(example, rating, batch_size, capacity, min_after_dequeue, shuffle=True, num_threads=4)
			example_batch, rating_batch = batch_graph_join(example_list, batch_size, capacity, min_after_dequeue, shuffle=True)
			print("expected: %s" % rating_batch)

	with tf.name_scope("ID_reshape"):
		with tf.device("/cpu:0"):
			#userId_batch = tf.reshape(example_batch[:, 0], shape=[-1])
			#movieId_batch = tf.reshape(example_batch[:, 1], shape=[-1])
			userId_batch = example_batch[:, 0]
			movieId_batch = example_batch[:, 1]

	with tf.name_scope("embedding_lookup"):
		#with tf.device("/device:gpu:0"):
		user_embedding_batch = tf.nn.embedding_lookup(user_embedding, userId_batch)
		tf.summary.histogram("user_embedding", user_embedding_batch)
		movie_embedding_batch = tf.nn.embedding_lookup(movie_embedding, movieId_batch)
		tf.summary.histogram("movie_embedding", movie_embedding_batch)

	with tf.name_scope("score"):
		similarity = tf.multiply(user_embedding_batch, movie_embedding_batch, name="similarity")
		print("similarity: %s" % similarity)
		w, b, pred = linear_block(similarity, embedding_width, 1, "linear_score")
		print("score: %s" % pred)

	with tf.name_scope("optimize"):
		#l1embedding = tf.contrib.layers.l1_regularizer(0.1)
		#tf.contrib.layers.apply_regularization(l1embedding, weights_list=[user_embedding, movie_embedding])
		#l2wb = tf.contrib.layers.l2_regularizer(0.01)
		#tf.contrib.layers.apply_regularization(l2wb, weights_list=[w])

		rating_batch = tf.reshape(rating_batch, shape=[tf.shape(rating_batch)[0], 1])
		diff = rating_batch - pred
		print("diff: %s" % diff)
		loss = tf.reduce_mean(tf.square(diff))
		print("loss %s" % loss)
		tf.summary.histogram("loss", loss)

		global_step = tf.Variable(0, trainable=False, name="global_step")
		start_learning_rate = 0.1 
		decay_steps = 640000
		decay_rate = 0.96
		learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True, name="learning_rate")
		#
		optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
		train_op = optimizer.minimize(loss, global_step=global_step)
	
	return train_op, loss, pred, rating_batch

def train():
	train_op, loss, pred, rating = net()

	config = tf.ConfigProto()
	'''
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allocator_type = 'BFC'
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	'''
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		writer = tf.summary.FileWriter("./logs/", graph=sess.graph)
		loss_summary = tf.summary.scalar("loss", loss)
		summary = tf.summary.merge_all()

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer()) # local epoch count
		## multi-thread
		coord = tf.train.Coordinator() 
		threads = tf.train.start_queue_runners(sess, coord)
		batch_sn = 0
		diff = 0
		batch_cnt = 0
		try:
			while not coord.should_stop():
				batch_cnt += 1
				if batch_cnt % 100 == 0:
					batch_sn += batch_cnt
					#(op_, loss_''', summary_''') = sess.run((train_op, loss''', loss_summary'''))
					(op_, loss_, summary_) = sess.run((train_op, loss, summary))
					epoch_sn = (batch_sn * batch_size) // (100000)
					print("epoch:%d, examples:%d, loss = %f" % (epoch_sn, batch_sn * batch_size, loss_))
					writer.add_summary(summary_, batch_sn)
					batch_cnt = 0
				else:
					sess.run(train_op)
				
		except tf.errors.OutOfRangeError:
			print("Done! Now kill all the threads.")
		finally:
			coord.request_stop()
			print("Ask all threads to stop.")
		coord.join(threads)
		writer.close()
	
def get_argv():
	parse = argparse.ArgumentParser()
	parse.add_argument("--filename", metavar='N', type=str, nargs='+', default="./ratings.csv", help="Csv file name")
	parse.add_argument("--testsize", type=int, default=1000, help="Size of test set")

	script = sys.argv[0]
	tag, unparsed = parse.parse_known_args(sys.argv[1:])
	print(tag)
	return tag, unparsed

def maxIds(filename, userid=0, movieid=0):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		fields = next(reader)
		for row in reader:
			#print(row)
			userid = int(row[0]) if int(row[0]) > userid else userid
			movieid = int(row[1]) if int(row[1]) > movieid  else movieid
	return userid, movieid 

###############################################

def main():
	tag, _ = get_argv()
	global filenames
	filenames = tag.filename

	global max_userId, max_movieId
	for filename in filenames:
		print(filename)
		max_userId, max_movieId = maxIds(filename, max_userId, max_movieId)

	print("max_user_id:%d, max_movie_id:%d" % (max_userId, max_movieId))
	train()

if __name__ == '__main__':
	main()


