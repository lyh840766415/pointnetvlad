import numpy as np
import tensorflow as tf
from pointnetvlad_cls import *
from loading_pointclouds import *


#Global Variable 
BATCH_NUM_QUERIES = 2
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_QUERY = 18
EPOCH = 100
NUM_POINTS = 4096
GPU_INDEX = 1
TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
TEST_FILE = 'generating_queries/test_queries_baseline.pickle'
LOG_DIR = "log/"
TRAINING_QUERIES= get_queries_dict(TRAIN_FILE)
TEST_QUERIES= get_queries_dict(TEST_FILE)

#batch norm parameter
DECAY_STEP = 20000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

#loss parameter
MARGIN1 = 0.5
MARGIN2 = 0.2

#learning_rate_parameter
BASE_LEARNING_RATE = 0.00005
is_training = True
OPTIMIZER = "adam"

def get_bn_decay(batch):
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,batch*BATCH_NUM_QUERIES,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay
	
#learning rate halfed every 5 epoch
def get_learning_rate(epoch):
	learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def init_network():
	with tf.device('/gpu:'+str(GPU_INDEX)):
		print("In Graph")
		query= placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
		positives= placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
		negatives= placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
		other_negatives=  placeholder_inputs(BATCH_NUM_QUERIES,1, NUM_POINTS)

		is_training_pl = tf.placeholder(tf.bool, shape=())
		print(is_training_pl)
		
		batch = tf.Variable(0)
		epoch_num = tf.placeholder(tf.float32, shape=())
		bn_decay = get_bn_decay(batch)
		tf.summary.scalar('bn_decay', bn_decay)

		with tf.variable_scope("query_triplets") as scope:
			vecs= tf.concat([query, positives, negatives, other_negatives],1)
			print(vecs)
			out_vecs= forward(vecs, is_training_pl, bn_decay=bn_decay)
			print(out_vecs)
			q_vec, pos_vecs, neg_vecs, other_neg_vec= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY,1],1)
			print(q_vec)
			print(pos_vecs)
			print(neg_vecs)
			print(other_neg_vec)

		#loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN1)
		#loss = softmargin_loss(q_vec, pos_vecs, neg_vecs)
		#loss = quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
		loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
		tf.summary.scalar('loss', loss)

		# Get training operator
		learning_rate = get_learning_rate(epoch_num)
		tf.summary.scalar('learning_rate', learning_rate)
		if OPTIMIZER == 'momentum':
			optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
		elif OPTIMIZER == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss, global_step=batch)
		
		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()

	# Create a session
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	
	sess = tf.Session(config=config)

	# Add summary writers
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

	# Initialize a new model
	init = tf.global_variables_initializer()
	sess.run(init)
	print("Initialized")
	
	#prepare the fid_dict
	ops = {'query': query,
		'positives': positives,
		'negatives': negatives,
		'other_negatives': other_negatives,
		'is_training_pl': is_training_pl,
		'loss': loss,
		'train_op': train_op,
		'merged': merged,
		'step': batch,
		'epoch_num': epoch_num,
		'q_vec':q_vec,
		'pos_vecs': pos_vecs,
		'neg_vecs': neg_vecs,
		'other_neg_vec': other_neg_vec,
		'sess': sess,
		'train_writer': train_writer,
		'test_writer': test_writer}
	
	return ops
	
	
def select_batch(train_file_idxs,batch_ind):
	batch_keys= train_file_idxs[batch_ind*BATCH_NUM_QUERIES:(batch_ind+1)*BATCH_NUM_QUERIES]
	q_tuples=[]
	
	print("batch %d"%(batch_ind))
	for j in range(BATCH_NUM_QUERIES):
		if(len(TRAINING_QUERIES[batch_keys[j]]["positives"])<POSITIVES_PER_QUERY):
			break
		
		#choose some other neg, what is definition of other neg
		q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
		
	queries=[]
	positives=[]
	negatives=[]
	other_neg=[]
	
	for k in range(len(q_tuples)):
		queries.append(q_tuples[k][0])
		positives.append(q_tuples[k][1])
		negatives.append(q_tuples[k][2])
		other_neg.append(q_tuples[k][3])
		
	#convert to numpy array
	queries= np.array(queries)
	queries= np.expand_dims(queries,axis=1)
	other_neg= np.array(other_neg)
	other_neg= np.expand_dims(other_neg,axis=1)
	positives= np.array(positives)
	negatives= np.array(negatives)
	return queries, positives,negatives,other_neg
	

def main():
	#arguments define as the global variable
	
	#init network
	ops = init_network()
				
	#shuffle train files
	for ep in range(EPOCH):
		train_file_idxs = np.arange(0,len(TRAINING_QUERIES.keys()))
		np.random.shuffle(train_file_idxs)
		print('train_file_uum = %f , BATCH_NUM_QUERIES = %f , iteration per batch = %f' %(len(train_file_idxs), BATCH_NUM_QUERIES,len(train_file_idxs)//BATCH_NUM_QUERIES))
		
		for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
			queries,positives,negatives,other_neg = select_batch(train_file_idxs,i)
			
			#send the batch to the network
			feed_dict={ops['query']:queries, ops['positives']:positives, ops['negatives']:negatives, ops['other_negatives']:other_neg, ops['is_training_pl']:is_training, ops['epoch_num']:ep}
							
			#train this batch
			summary, step, train, loss_val = ops['sess'].run([ops['merged'], ops['step'],ops['train_op'], ops['loss']], feed_dict=feed_dict)
			
			
			#log something accuracy recall ......
			ops['train_writer'].add_summary(summary, step)
			print('batch loss: %f' %(loss_val))



if __name__ == '__main__':
	main()