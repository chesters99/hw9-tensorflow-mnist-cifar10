import argparse
import numpy as np
import os
import sys
from datetime import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

FLAGS = None


def train(_):
    # create new log files
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    # Import data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    X_train = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    batch_size = 500

    gen = ImageDataGenerator(rotation_range=6, width_shift_range=0.06, shear_range=0.27,
                             height_shift_range=0.06, zoom_range=0.06)
    train_gen = gen.flow(X_train, y_train, batch_size=batch_size, seed=0)

    # Create a multilayer model.
    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # def bias_variable(shape):
    #     """Create a bias variable with appropriate initialization."""
    #     initial = tf.constant(0.1, shape=shape)
    #     return tf.Variable(initial)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
            # with tf.name_scope('biases'):
            #     biases = bias_variable([output_dim])
            with tf.name_scope('Wx_plus_b'):
                preact = tf.matmul(input_tensor, weights) # + biases
                tf.summary.histogram('pre_activations', preact)

            # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
            with tf.name_scope('batch_norm'): 
                b_mean, b_var = tf.nn.moments(preact, [0])
                scale = tf.Variable(tf.ones([output_dim]))
                beta  = tf.Variable(tf.zeros([output_dim]))
                preact = tf.nn.batch_normalization(preact, b_mean, b_var, beta, scale, 1e-3)

            activations = act(preact, name='activation')
            tf.summary.histogram('activations', activations)
            return activations



    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=y_, logits=y)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph,flush_secs=10)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test',flush_secs=10)
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train:
            xs, ys = next(train_gen)
            xs = xs.reshape(batch_size, 28*28)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps+1):
        if i % 100 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('%s: Accuracy at step %s: %s' % (datetime.now(), i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions()
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                # print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                if i % 10 == 0:
                    train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=10000,   # 1000
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tf/mnist/input_data'),
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tf/mnist/augm_bn'),
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
