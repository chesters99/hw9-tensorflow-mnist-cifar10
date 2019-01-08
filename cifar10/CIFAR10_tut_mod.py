import argparse
import numpy as np
import os
import sys
import pickle
from datetime import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train(_):
    # create new log files
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    batch_size = 200
    num_classes = 10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print("Train Images dimensions =", X_train.shape)
    print("Train Labels dimensions =", y_train.shape)
    print("Test Images dimensions =", X_test.shape)
    print("Test Labels dimensions =", y_test.shape)

    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    the_mean = X_train.mean(axis=0)
    the_std  = X_train.std(axis=0)
    X_train = (X_train - the_mean) / the_std
    X_test  = (X_test  - the_mean) / the_std

    label_list_path = '/Users/graham/.keras/datasets/cifar-10-batches-py/batches.meta'
    with open(label_list_path, mode='rb') as f:
        labels = pickle.load(f)
    labels = labels['label_names']

    gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.15, shear_range=0.15,
                             height_shift_range=0.15, zoom_range=0.15, horizontal_flip=True,) 
    train_gen = gen.flow(X_train, y_train, batch_size=batch_size, seed=0)

    # Create a multilayer model.
    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32, [None, 32*32*3], name='x-input')
        y_ = tf.placeholder(tf.int64,   [None, num_classes], name='y-input')

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.05) # 0.1
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2dx(x, num_outputs):
        return tf.contrib.layers.conv2d(x, kernel_size=[5,5], num_outputs=num_outputs,
            stride=[1, 1], padding='SAME',
            # normalizer_fn=tf.layers.batch_normalization,
            # normalizer_params={"training": False, "reuse": False},
            activation_fn=tf.nn.elu,)


    x_image = tf.reshape(x, [-1,32,32,3])

    # conv1 
    h_conv1 = conv2dx(x_image, 64)
    print ("h_conv1" + str(h_conv1.get_shape()))

    # pool1
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    print ("h_pool1" + str(h_pool1.get_shape()))

    # norm1
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

    # conv2 
    h_conv2 = conv2dx(h_norm1, 64)
    print ("h_conv2" + str(h_conv2.get_shape()))

    # norm2
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    #pool2
    h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    print ("h_pool2" + str(h_pool2.get_shape()))

    #flatten
    h_flat = tf.reshape(h_pool2, [-1, 32*32*4])
    print ("h_flat" + str(h_flat.get_shape()))

    # fully connected 1
    W_fc1 = weight_variable([32*32*4, 384])
    b_fc1 = bias_variable([384])
    h_fc1 = tf.nn.elu(tf.matmul(h_flat, W_fc1) + b_fc1)
    print ("h_fc1" + str(h_fc1.get_shape()))

    # fully connected 2
    W_fc2 = weight_variable([384, 192])
    b_fc2 = bias_variable([192])
    h_fc2 = tf.nn.elu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    print ("h_fc2" + str(h_fc2.get_shape()))

    # fully connected 3 then softmax the output
    W_fc3 = weight_variable([192, 10])
    b_fc3 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
    print ("y" + str(y.get_shape()))

    keep_prob = tf.placeholder(tf.float32) # not used

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(
                tf.cast(y_, tf.float32) * tf.log(y+1e-10), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1)) 
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph,flush_secs=10)
    test_writer  = tf.summary.FileWriter(FLAGS.log_dir + '/test',flush_secs=10)
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train:
            xs, ys = next(train_gen)
            xs = xs.reshape(batch_size, 32*32*3)
            k = 0.25
        else:
            xs, ys = X_test, y_test
            xs = xs.reshape(10000, 32*32*3)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    print('Start Training:', datetime.now())
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
    parser.add_argument('--max_steps', type=int, default=80000,   # 1000
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.75, # not used
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tf/cifar10/input_data'),
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tf/cifar10/tut_modelu'),
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
