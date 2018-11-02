import tensorflow as tf
import numpy as np

a = np.arange(40)
print a

b = np.reshape(a, (2,4,5))
print b

c = tf.transpose(b,[0,1,2])
sess = tf.Session()
print sess.run(c)

d = tf.transpose(b, [1,0,2])
print sess.run(d)

cun = tf.unstack(c )
print sess.run(cun)

dun = tf.unstack(d)
print sess.run(dun)