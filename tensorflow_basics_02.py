################################################################################################
#	name:	tensorflow_basics_02.py
#	desc:	Gettig started with tensorflow with simple multiplication of matrices
#	date:	2019-01-19
#	Author:	conquistadorjd
################################################################################################
import tensorflow as tf

print("*** Program Started ***")
# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)

with tf.Session() as sess:
  output = sess.run(result)
  print(output)

print("*** Program Ended ***")