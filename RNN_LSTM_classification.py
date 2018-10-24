#RNN 来进行分类的训练 (Classification). 会继续使用到手写数字 MNIST 数据集. 让 RNN 从每张图片的第一行像素读到最后一行, 然后再进行分类判断

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.set_random_seed(1) #set random seed

lr=0.001  # learning rate
training_iters=100000 # train step 上限
batch_size=128
n_inputs=28 # MNIST data input (img shape: 28*28)
n_steps=28 #time steps
n_hidden_units=128  #neurons in hidden layer
n_classes=10  #MNIST classes (0-9 digits)

x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

weights={'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
		 'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))}

biases={'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
	     'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))}

def RNN(X,weights,biases):
	X=tf.reshape(X,[-1,n_inputs])      # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
                                       # X ==> (128 batches * 28 steps, 28 inputs)
	X_in=tf.matmul(X,weights['in'])+biases['in']
	X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])     # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
	
	# 使用 basic LSTM Cell.
	lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
	init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
	outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
	
	results=tf.matmul(final_state[1],weights['out'])+biases['out']
	return results


pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)


correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step=0
	while step * batch_size<training_iters:
		batch_xs,batch_ys=mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
		sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
		if step % 20 == 0:
			print(sess.run(accuracy, feed_dict={
			x: batch_xs,
			y: batch_ys,
		}))
		step += 1

















