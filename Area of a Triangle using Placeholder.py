import tensorflow as tf

def compute_area(sides):
  a = sides[:, 0]           # 1.5 and 3.5
  b = sides[:, 1]           # 2.5 and 2.5
  c = sides[:, 2]           # 3.5 and 1.5
  S = (a + b + c)/2
  divident = S*(S - a)*(S - b)*(S - c)
  return tf.sqrt(divident)

sides = tf.placeholder("float32", shape=(None, 3))
with tf.Session() as session:
  area = compute_area(sides)
  result = session.run(area, feed_dict={
      sides: [[1.5, 2.5, 3.5],
              [3.5, 2.5, 1.5]] 
  })
  print(result)
