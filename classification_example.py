from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # MNIST 数据

import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
      outputs=Wx_plus_b
    else:
      outputs=activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs=tf.placeholder(tf.float32,[None,784]) # 每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据
ys=tf.placeholder(tf.float32,[None,10]) #每张图片都表示一个数字，所以我们的输出是数字0到9，共10类

prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax) 
# 输入数据是784个特征，输出数据是10个特征，激励采用softmax函数

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
# loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # 采用梯度下降法

sess=tf.Session()
sess.run(tf.global_variables_initializer())

batch_xs,batch_ys=mnist.train.next_batch(100)
sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
