# Tensorflow example 

1.TF_placeholder_example.py  
	如果想要从外部传入data, 那就需要用到 tf.placeholder(),然后以这种形式传输数据 sess.run(***, feed_dict={input: **})

2.TF_scope_example.py  
	tf.get_variable() # the gobal variables  
	tf.Variable() # the variables with scope

3.TF_session_example.py 
	sess=tf.Session()/with tf.Session() as sess

4.TF_example.py 
	y=0.1*x + 0.3	
	using one nn node to fitting the parameters

5.TF_example_3.py
	我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络
	fitting: y=x^2+0.5

6.CNN_minist.py 

	https://zhuanlan.zhihu.com/p/51322893




7.classification_minist.py
	分类MNIST库
	loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零

8.RNN_LSTM_classification.py
	处理序列数据的神经网络
	LSTM 是 long-short term memory 的简称, 中文叫做 长短期记忆. 是当下最流行的 RNN 形式之一

9.RNN_LSTM_regression.py
