import tensorflow as tf

def compute_area(sides):
  a = sides[:, 0]
  b = sides[:, 1]
  c = sides[:, 2]
  S = (a + b + c)/2
  divident = S*(S - a)*(S - b)*(S - c)
  return tf.sqrt(divident)

with tf.Session() as session:
  area = compute_area(tf.constant([[1.5, 2.5, 3.5],
                                   [3.5, 2.5, 1.5]]))
  print(session.run(area))
