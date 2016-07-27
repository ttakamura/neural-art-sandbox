import tensorflow as tf
import numpy as np
from utils import read_image, save_image, parseArgs, getModel, add_mean
import time

def get_feature_vectors(sess, model, image):
  image_x      = tf.constant(image)
  image_y_vals = [sess.run(y_l, {model.input_tensor: image_x}) for y_l in model.y()]
  return image_y_vals

def get_style_matrixes(sess, y, num_filters):
  image_st_vals = []
  for l in range(len(y)):
    num_filter = num_filters[l]
    st_ = tf.reshape(y[l], [-1, num_filter])
    st  = tf.matmul(tf.transpose(st_), st_)
    image_st_vals.append(sess.run(st))  # sess.run(st) is a constant numpy array
  return image_st_vals

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('content_image_path', '/tmp/hoge.jpg',    'Content Image path')
tf.app.flags.DEFINE_string('style_image_path',   '/tmp/style.jpg',   'Style Image path')
tf.app.flags.DEFINE_string('graph_path',         '/tmp/imagenet.pb', 'TensorFlow graph path')
tf.app.flags.DEFINE_integer('iters',             100,                'Num of iterations')

alpha        = 1.0
beta         = 1.0
modeltype    = 'inception'
expect_width = None

# The actual calculation
print "Read images..."
content_image = read_image(FLAGS.content_image_path, expect_width)
style_image   = read_image(FLAGS.style_image_path,   expect_width)

with tf.Session() as sess:
  model = getModel(sess, FLAGS.graph_path, modeltype)

  print "Load content values..."
  content_image_y_vals = get_feature_vectors(sess, model, content_image)
  num_filters          = [y.shape[3] for y in content_image_y_vals]
  print([y.shape for y in content_image_y_vals])

  print "Load style values..."
  style_image_y_vals  = get_feature_vectors(sess, model, style_image)
  style_image_st_vals = get_style_matrixes(sess, style_image_y_vals, num_filters)
  print([y.shape for y in style_image_st_vals])

  print "Construct graph..."
  # Start from white noise
  # gen_image = tf.Variable(tf.truncated_normal(content_image.shape, stddev=20), trainable=True, name='gen_image')
  # Start from the original image
  gen_image = tf.Variable(tf.constant(np.array(content_image, dtype=np.float32)), trainable=True, name='gen_image')

    model = getModel(gen_image, FLAGS.graph_path, modeltype)
    y = model.y()
    L_content = 0.0
    L_style   = 0.0
    for l in range(len(y)):
        # Content loss
        L_content += model.alpha[l]*tf.nn.l2_loss(y[l] - content_image_y_val[l])
        # Style loss
        num_filters = content_image_y_val[l].shape[3]
        st_shape = [-1, num_filters]
        st_ = tf.reshape(y[l], st_shape)
        st = tf.matmul(tf.transpose(st_), st_)
        N = np.prod(content_image_y_val[l].shape).astype(np.float32)
        L_style += model.beta[l]*tf.nn.l2_loss(st - style_image_st_val[l])/N**2/len(y)

    # The loss
    L = alpha* L_content + beta * L_style
    # The optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(L, global_step=global_step)
    # A more simple optimizer
    # train_step = tf.train.AdamOptimizer(learning_rate=2.0).minimize(L)

    # Set up the summary writer (saving summaries is optional)
    # (do `tensorboard --logdir=/tmp/na-logs` to view it)
    tf.scalar_summary("L_content", L_content)
    tf.scalar_summary("L_style", L_style)
    gen_image_addmean = tf.Variable(tf.constant(np.array(content_image, dtype=np.float32)), trainable=False)
    tf.image_summary("Generated image (TODO: add mean)", gen_image_addmean)
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/na-logs', graph_def=sess.graph_def)

    print "Start calculation..."
    # The optimizer has variables that require initialization as well
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.iters):
        if i % 10 == 0:
            gen_image_val = sess.run(gen_image)
            save_image(gen_image_val, i, args.out_dir)
            print "L_content, L_style:", sess.run(L_content), sess.run(L_style)
            # Increment summary
            sess.run(tf.assign(gen_image_addmean, add_mean(gen_image_val)))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)
        print "Iter:", i
        sess.run(train_step)
