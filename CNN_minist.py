import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)  #采用的数据集依然是tensorflow里面的mnist数据集

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict = {xs : v_xs, keep_prob : 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs : v_xs, ys : v_ys, keep_prob : 1})
    return result

# define weight,Weight变量，输入shape，返回变量的参数。其中我们使用tf.truncted_normal产生随机变量来进行初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
# define biases,# 定义biase变量，输入shape ,返回变量的一些参数。其中我们使用tf.constant常量函数来进行初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# cov2d 步长 [1, 1] 
#tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，
#中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# max_pool
#接着定义池化pooling，为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，
#因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()
#池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
def max_poo_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# define placeholder for inputting data
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# define placeholder for dropout
# 定义了dropout的placeholder，它是解决过拟合的有效手段
keep_prob = tf.placeholder(tf.float32)

# 把数据转换成 适合 nn 输入的数据格式 ，-1代表先不考虑输入的图片例子多少个维度
#把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# 1.convolutional layer1 + max pooling;
# 2.convolutional layer2 + max pooling;
# 3.fully connected layer1 + dropout;
# 4.fully connected layer2 to prediction.

# 定义第一层卷积
#先定义本层的Weight,本层我们的卷积核patch的大小是5x5,因为黑白图片channel是1所以输入是1，输出是32个featuremap
#定义bias，它的大小是32个长度，因此我们传入它的shape为[32]
 #第一个卷积层,同时我们对h_conv1进行非线性处理，也就是激活函数来处理,因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32
 #进行pooling的处理就ok啦，经过pooling的处理，输出大小就变为了14x14x32

# patch 5x5 ,channel is 1 ， output 32 featuremap
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# structure is 28x28x32
h_cov1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# structure is 14x14x32
h_pool1 = max_poo_2x2(h_cov1)



#定义第二层卷积
 #卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出我们定为64

  #定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64
#pooling处理，输出大小为7x7x64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# structure is 14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# structure is 7x7x64
h_pool2 = max_poo_2x2(h_conv2)

# full connection
#建立全连接层

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#建立全连接层
W_fc1 = weight_variable([7*7*64, 1024]) ## 后面的输出size我们继续扩大，定为1024
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）

# add dropout  #考虑过拟合问题，可以加一个dropout的处理
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#进行最后一层的构建了，好激动啊, 输入是1024，最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值
# last layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# softmax 
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# define loss 
cross_entropy = tf.reduce_mean( - tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]) )  #利用交叉熵损失函数来定义我们的cost function

train_step = tf.train.AdamOptimizer( 1e-4 ).minimize( cross_entropy )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {xs : batch_xs, ys : batch_ys, keep_prob : 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
