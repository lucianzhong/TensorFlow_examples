import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist=input_data.read_data_sets('MNIST_data',one_hot=True)  #采用的数据集依然是tensorflow里面的mnist数据集


#Weight变量，输入shape，返回变量的参数。其中我们使用tf.truncted_normal产生随机变量来进行初始化
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_varibale(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，
#然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，
#中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME
def conv2d(x,W):
    return tf.nn.con2d(x,W,strides=[1,1,1,1],padding='SAME')

#接着定义池化pooling，为了得到更多的图片信息，padding时我们选的是一次一步，
#也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，
#而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，
#因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。
#pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()
#池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:

def max_poo_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1])

# 1.convolutional layer1 + max pooling;
# 2.convolutional layer2 + max pooling;
# 3.fully connected layer1 + dropout;
# 4.fully connected layer2 to prediction.

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

keep_prob=tf.placeholder(tf.float32) # 定义了dropout的placeholder，它是解决过拟合的有效手段

x_image=tf.reshape(xs,[-1,28,28,1]) 
#把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
#因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。


# 定义第一层卷积
W_conv1=weight_variable([5,5,1,32]) #先定义本层的Weight,本层我们的卷积核patch的大小是5x5，

#因为黑白图片channel是1所以输入是1，输出是32个featuremap

b_conv1=bias_varibale([32])  #定义bias，它的大小是32个长度，因此我们传入它的shape为[32]



























