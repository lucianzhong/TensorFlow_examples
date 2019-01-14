import tensorflow as tf
import numpy as np
import matplotlib as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
      outputs=Wx_plus_b
    else:
      outputs=activation_function(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise # 构建所需的数据


xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1]) 
#利用占位符定义我们所需的神经网络的输入,tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1	


#通常神经层都包括输入层、隐藏层和输出层。这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，我们假设隐藏层有10个神经元； 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。
#所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。

#  Tensorflow 自带的激励函数tf.nn.relu。
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# 定义输出层。此时的输入就是隐藏层的输出l1，输入有10层（隐藏层的输出层），输出有1层。
prediction=add_layer(l1,10,1,activation_function=None)
# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
# train method
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 使用变量时，都要对它进行初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)  # 在tensorflow中，只有session.run()才会执行我们定义的运算

# 我们让机器学习1000次。机器学习的内容是train_step, 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。
# (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入
for i in range(100000):
 sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
 if i%50==0:
   print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
   










