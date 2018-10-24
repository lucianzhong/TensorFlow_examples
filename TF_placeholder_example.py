import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

sess=tf.Session()
print(sess.run(output,feed_dict={input1:[7], input2:[2]}))


#如果想要从外部传入data, 那就需要用到 tf.placeholder(), 
#然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).