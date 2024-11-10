import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
NUM_CLASSES = 1000

def fire_module(x, inp, sp, e11p, e33p):
    with tf.compat.v1.variable_scope("fire",reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope("squeeze"):
            W = tf.compat.v1.get_variable("weights", shape=[1, 1, inp, sp])
            b = tf.compat.v1.get_variable("bias", shape=[sp])
            s = tf.compat.v1.nn.conv2d(x, W, [1, 1, 1, 1], "VALID") + b
            s = tf.compat.v1.nn.relu(s)
        with tf.compat.v1.variable_scope("e11"):
            W = tf.compat.v1.get_variable("weights", shape=[1, 1, sp, e11p])
            b = tf.compat.v1.get_variable("bias", shape=[e11p])
            e11 = tf.compat.v1.nn.conv2d(s, W, [1, 1, 1, 1], "VALID") + b
            e11 = tf.compat.v1.nn.relu(e11)
        with tf.compat.v1.variable_scope("e33"):
            W = tf.compat.v1.get_variable("weights", shape=[3, 3, sp, e33p])
            b = tf.compat.v1.get_variable("bias", shape=[e33p])
            e33 = tf.compat.v1.nn.conv2d(s, W, [1, 1, 1, 1], "SAME") + b
            e33 = tf.compat.v1.nn.relu(e33)
        return tf.compat.v1.concat([e11, e33], axis=3)

class SqueezeNet(object):
    def __init__(self, save_path=None, sess=None):
        self.image = tf.compat.v1.placeholder('float', shape=[None, None, None, 3], name='input_image')
        self.labels = tf.compat.v1.placeholder('int32', shape=[None], name='labels')
        self.layers = self.extract_features(self.image, reuse=False)
        self.features = self.layers[-1]
        self.classifier = self.build_classifier(self.features)

        if save_path is not None and sess is not None:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, save_path)

        self.loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), logits=self.classifier))

    def extract_features(self, input, reuse=True):
        x = input
        layers = []
        with tf.compat.v1.variable_scope('features', reuse=reuse):
            x = self.conv_layer(x, [3, 3, 3, 64], 64, 'layer0', stride=2)
            x = tf.compat.v1.nn.relu(x)
            layers.append(x)
            x = tf.compat.v1.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            layers.append(x)
            x = fire_module(x, 64, 16, 64, 64)
            layers.append(x)
            x = fire_module(x, 128, 16, 64, 64)
            layers.append(x)
            x = tf.compat.v1.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            layers.append(x)
            x = fire_module(x, 128, 32, 128, 128)
            layers.append(x)
            x = fire_module(x, 256, 32, 128, 128)
            layers.append(x)
            x = tf.compat.v1.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            layers.append(x)
            x = fire_module(x, 256, 48, 192, 192)
            layers.append(x)
            x = fire_module(x, 384, 48, 192, 192)
            layers.append(x)
            x = fire_module(x, 384, 64, 256, 256)
            layers.append(x)
            x = fire_module(x, 512, 64, 256, 256)
            layers.append(x)
        return layers

    def conv_layer(self, x, filter_shape, num_filters, scope, stride=1):
        with tf.compat.v1.variable_scope(scope):
            W = tf.compat.v1.get_variable("weights", shape=filter_shape)
            b = tf.compat.v1.get_variable("bias", shape=[num_filters])
            x = tf.compat.v1.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
            return tf.compat.v1.nn.bias_add(x, b)

    def build_classifier(self, features):
        with tf.compat.v1.variable_scope('classifier'):
            W = tf.compat.v1.get_variable("weights", shape=[1, 1, 512, NUM_CLASSES])
            b = tf.compat.v1.get_variable("bias", shape=[NUM_CLASSES])
            x = tf.compat.v1.nn.conv2d(features, W, [1, 1, 1, 1], "VALID") + b
            x = tf.compat.v1.nn.relu(x)
            x = tf.compat.v1.nn.avg_pool(x, [1, 13, 13, 1], strides=[1, 13, 13, 1], padding='VALID')
            return tf.compat.v1.reshape(x, [-1, NUM_CLASSES])