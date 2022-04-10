import tensorflow as tf
import utils
import sys


EPS = 1e-10

# command arguments
phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_psnr, w_gan, w_color, w_contrast, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)


def resnet(input_image, batch_size = batch_size):
    with tf.variable_scope("generator"):
        def luma_stream(luma_input):
            with tf.variable_scope("luma_stream"):
                # [batch, 100, 100, 1]  => [batch, 100, 100, 64]
                W1 = weight_variable([9, 9, 1, 64], name="W1");
                b1 = bias_variable([64], name="b1");
                c1 = tf.nn.sigmoid(conv2d(luma_input, W1, strides=[1, 1, 1, 1]) + b1)

                # [batch, 100, 100, 64]  => [batch, 50, 50, 64]
                W1_2 = weight_variable([3, 3, 64, 128], name="W1_2");
                b1_2 = bias_variable([128], name="b2");
                c1_2 = tf.nn.sigmoid(_instance_norm(conv2d(c1, W1_2, strides=[1, 2, 2, 1])) + b1_2)

                # [batch, 50, 50, 64] => [batch, 25, 25, 128]
                W1_3 = weight_variable([3, 3, 128, 256], name="W1_3");
                b1_3 = bias_variable([256], name="b3");
                c1_3 = tf.nn.sigmoid(_instance_norm(conv2d(c1_2, W1_3, strides=[1, 2, 2, 1])) + b1_3)
            return [c1, c1_2, c1_3]

        input_image_ = input_image[:, :, :, 0] * 0.299 + \
                       input_image[:, :, :, 1] * 0.587 + \
                       input_image[:, :, :, 2] * 0.114
        input_image_Y = tf.ones_like(input_image_) - input_image_
        w = input_image_.shape[1]
        h = input_image_.shape[2]
        attention = tf.reshape(input_image_Y, shape=[-1, w, h, 1])
        [Y_c1, Y_c1_2, Y_c1_3] = luma_stream(attention)
        # [batch, 100, 100, 3]  => [batch, 100, 100, 64]
        input_ = tf.concat([input_image, attention], axis=3)
        W1 = weight_variable([9, 9, 4, 64], name="W1"); b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(input_, W1, strides=[1, 1, 1, 1]) + b1)
        # c1 = tf.concat([c1, Y_c1], axis=3)
        c1 = c1 * Y_c1 + c1

        # [batc2h, 100, 100, 64]  => [batch, 50, 50, 64]
        W1_2 = weight_variable([3, 3, 64, 128], name="W1_2"); b1_2 = bias_variable([128], name="b1_2");
        c1_2 = tf.nn.relu(_instance_norm(conv2d(c1, W1_2, strides=[1, 2, 2, 1])) + b1_2)
        # c1_2 = tf.concat([c1_2, Y_c1_2], axis=3)
        # c1_2 = c1_2 + Y_c1_2
        c1_2 = c1_2 * Y_c1_2 + c1_2

        # [batch, 50, 50, 64] => [batch, 25, 25, 128]
        W1_3 = weight_variable([3, 3, 128, 256], name="W1_3");b1_3 = bias_variable([256], name="b1_3");
        c1_3 = tf.nn.relu(_instance_norm(conv2d(c1_2, W1_3, strides=[1, 2, 2, 1])) + b1_3)
        # c1_3 = tf.concat([c1_3, Y_c1_3], axis=3)
        c1_3 = c1_3 * Y_c1_3 + Y_c1_3

        # global_feature block

        # [batch, 25, 25, 256] => [batch, 13, 13, 256]
        W1_4 = weight_variable([5, 5, 256, 256], name="W1_4");
        c1_4 = conv2d(c1_3, W1_4, strides=[1, 2, 2, 1])

        # [batch, 13, 13, 256] => [batch, 7, 7, 256]
        W1_5 = weight_variable([5, 5, 256, 256], name="W1_5");
        c1_5 = conv2d(c1_4, W1_5, strides=[1, 2, 2, 1])

        # [batch, 7, 7, 256] => [batch, 4, 4, 256]
        W1_6 = weight_variable([5, 5, 256, 256], name="W1_6");
        c1_6 = conv2d(c1_5, W1_6, strides=[1, 2, 2, 1])

        # [batch, 4, 4, 256] => [batch, 2, 2, 256]
        W1_7 = weight_variable([3, 3, 256, 256], name="W1_7");
        c1_7 = conv2d(c1_6, W1_7, strides=[1, 2, 2, 1])

        # [batch, 2, 2, 256] => [batch, 1, 1, 256]
        c1_8 = tf.nn.max_pool(c1_7, [1, c1_7.get_shape().as_list()[1], c1_7.get_shape().as_list()[2], 1], [1, 1, 1, 1], padding='VALID')

        # [batch, 1, 1, 128] => [batch, 1, 1, 512]
        c1_8 = tf.layers.dense(c1_8, 512)

        # [batch, 1, 1, 512] => [batch, 1, 1, 256]
        c1_8 = tf.layers.dense(c1_8, 256)

        # # [batch, 1, 1, 256] => [batch, 1, 1, 128]
        # c1_8 = tf.layers.dense(c1_8, 128)


        # residual 1
        W2 = weight_variable([3, 3, 256, 256], name="W2");
        b2 = bias_variable([256], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1_3, W2) + b2))

        W3 = weight_variable([3, 3, 256, 256], name="W3");
        b3 = bias_variable([256], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1_3
        c3 = RCAB(c3, 8)
        # residual 2

        W4 = weight_variable([3, 3, 256, 256], name="W4");
        b4 = bias_variable([256], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 256, 256], name="W5");
        b5 = bias_variable([256], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3
        c5 = RCAB(c5, 8)

        # residual 3

        W6 = weight_variable([3, 3, 256, 256], name="W6");
        b6 = bias_variable([256], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 256, 256], name="W7");
        b7 = bias_variable([256], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5
        c7 = RCAB(c7, 8)

        # global_concat

        # [batch, 25, 25, 128] =>[batch, 25, 25, 256]
        c1_8 = global_concat(c7, c1_8, batch_size)


        # [batch, 25, 25, 256] => [batch, 25, 25, 128]
        W1_9 = weight_variable([3, 3, 512, 256], name="W1_9");
        b1_9 = bias_variable([256], name="b1_9");
        c1_9 = tf.nn.relu(_instance_norm(conv2d(c1_8, W1_9) + b1_9))
        # c1_9 = c1_9 * Y_c1_9

        # [batch, 25, 25, 256] =>[batch, 25, 25, 512]
        c9 = tf.nn.relu(tf.concat([c1_9, c1_3], axis=3));

        # [batch, 25, 25, 256] =>[batch, 50, 50, 64]
        W9_2 = deconv2d(c9, 128, 3); b9_2 = bias_variable([128], name="b9_2");
        c9_2 = tf.nn.relu(_instance_norm(W9_2) + b9_2);
        # c9_2 = c9_2 * Y_c9_2

        # [batch, 50, 50, 64] =>[batch, 50, 50, 128]
        c9_2 = tf.nn.relu(tf.concat([c9_2, c1_2], axis=3));

        # [batch, 50, 50, 128] => [batch, 100, 100, 64]
        W9_3 = deconv2d(c9_2, 64, 3); b9_3 = bias_variable([64], name="b9_3");
        c9_3 = tf.nn.relu(_instance_norm(W9_3) + b9_3);
        # c9_3 = c9_3 * Y_c9_3


        # Convolutional

        # [batch, 100, 100, 64] => [batch, 100, 100, 128]
        c9_3 = tf.nn.relu(tf.concat([c9_3, c1], axis=3));

        # [batch, 100, 100, 128] => [batch, 100, 100, 64]
        W10 = weight_variable([3, 3, 128, 128], name="W10"); b10 = bias_variable([128], name="b10");
        c10 = tf.nn.relu(conv2d(c9_3, W10) + b10)

        # [batch, 100, 100, 64] => [batch, 100, 100, 64]
        W11 = weight_variable([3, 3, 128, 64], name="W11"); b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Final

        # [batch, 100, 100, 64] => [batch, 100, 100, 3]
        W12 = weight_variable([9, 9, 64, 3], name="W12"); b12 = bias_variable([3], name="b12");
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced

def create_discriminator(discrim_inputs, discrim_targets, targets):
    # [batch, 100, 100, 3] => [batch, 100, 100, 6]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # [batch, 100, 100, 6] => [batch, 100, 100, 32]
    W1 = weight_variable([11, 11, 6, 32], name="W1");
    b1 = bias_variable([32], name="b1");
    c1 = lrelu(_instance_norm(conv2d(input, W1, strides=[1, 1, 1, 1], padding='VALID') + b1), 0.2)
    # print(c1.shape)

    # [batch, 100, 100, 32] => [batch, 50, 50, 64]
    W2 = weight_variable([7, 7, 32, 64], name="W2");
    b2 = bias_variable([64], name="b2");
    c2 = lrelu(_instance_norm(conv2d(c1, W2, strides=[1, 2, 2, 1], padding='VALID') + b2), 0.2)
    # print(c2.shape)

    # [batch, 50, 50, 64] => [batch, 25, 25, 128]
    W3 = weight_variable([5, 5, 64, 128], name="W3");
    b3 = bias_variable([128], name="b3");
    c3 = lrelu(_instance_norm(conv2d(c2, W3, strides=[1, 2, 2, 1], padding='VALID') + b3), 0.2)
    # print(c3.shape)

    # [batch, 25, 25, 128] => [batch, 12, 12, 128]
    W4 = weight_variable([5, 5, 128, 128], name="W4");
    b4 = bias_variable([128], name="b4");
    c4 = lrelu(_instance_norm(conv2d(c3, W4, strides=[1, 2, 2, 1], padding='VALID') + b4), 0.2)
    # print(c4.shape)

    # [batch, 12, 12, 128] => [batch, 12, 12, 1]
    W5 = weight_variable([5, 5, 128, 64], name="W5");
    b5 = bias_variable([64], name="b5");
    c5 = lrelu(_instance_norm(conv2d(c4, W5, strides=[1, 2, 2, 1], padding='VALID') + b5), 0.2)

    W6 = weight_variable([2, 2, 64, 1], name="W6");
    b6 = bias_variable([1], name="b6");
    c6 = tf.sigmoid(conv2d(c5, W6, strides=[1, 2, 2, 1], padding='VALID') + b6)

    dis_g = tf.nn.l2_loss(discrim_targets - targets)
    # local discriminator
    local_input = tf.concat([discrim_targets, targets], axis=3)
    local_ = tf.random_crop(local_input, size=(batch_size, 32, 32, 6))
    dis_l = tf.nn.l2_loss(local_[:, :, :, 0:2] - local_[:, :, :, 3:5])
    weight_g = (dis_g + EPS) / (dis_g + dis_l + EPS)
    weight_l = (dis_l + EPS) / (dis_g + dis_l + EPS)
    output_local = []
    for k in range(8):
        local_inputs = tf.random_crop(input, size=(batch_size, 32, 32, 6))
        # [batch, 32, 32, 6] => [batch, 32, 32, 32]
        W1_l = weight_variable([3, 3, 6, 32], name=str(k) + "/W1_l")
        b1_l = bias_variable([32], name=str(k) + "/b1_l")
        c1_l = lrelu(_instance_norm(conv2d(local_inputs, W1_l, strides=[1, 1, 1, 1], padding='VALID') + b1_l), 0.2)

        # [batch, 32, 32, 32] => [batch, 16, 16, 64]
        W2_l = weight_variable([3, 3, 32, 64], name=str(k) + "/W2_l")
        b2_l = bias_variable([64], name=str(k) + "/b2_l")
        c2_l = lrelu(_instance_norm(conv2d(c1_l, W2_l, strides=[1, 2, 2, 1], padding='VALID') + b2_l), 0.2)

        # [batch, 16, 16, 64] => [batch, 8, 8, 128]
        W3_l = weight_variable([3, 3, 64, 128], name=str(k) + "/W3_l")
        b3_l = bias_variable([128], name=str(k) + "/b3_l")
        c3_l = lrelu(_instance_norm(conv2d(c2_l, W3_l, strides=[1, 2, 2, 1], padding='VALID') + b3_l), 0.2)

        # [batch, 8, 8, 128] => [batch, 4, 4, 1]
        W5_l = weight_variable([3, 3, 128, 64], name=str(k) + "/W5_l")
        b5_l = bias_variable([64], name=str(k) + "/b5_l")
        c5_l = lrelu(_instance_norm(conv2d(c3_l, W5_l, strides=[1, 2, 2, 1], padding='VALID') + b5_l), 0.2)

        W6_l = weight_variable([2, 2, 64, 1], name=str(k) + "/W6_l")
        b6_l = bias_variable([1], name=str(k) + "/b6_l")
        c6_l = tf.sigmoid(conv2d(c5_l, W6_l, strides=[1, 2, 2, 1], padding='VALID') + b6_l)
        output_local.append(c6_l)
    output_local_p = tf.reduce_mean(output_local, axis=0)
    output = weight_g * c6 + weight_l * output_local_p
    return output

def RCAB(input, reduction):
    (batch, height, width, channel) = input.shape
    f = tf.layers.conv2d(input, channel, 3, padding='same', activation=tf.nn.relu)
    f = tf.layers.conv2d(f, channel, 3, padding='same')

    x = tf.reduce_mean(f, axis=(1, 2), keep_dims=True)
    x = tf.layers.conv2d(x, channel // reduction, 1, activation=tf.nn.relu)
    x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)
    x = tf.multiply(f, x)
    x = tf.add(input, x)
    return x


def weight_variable(shape, name):
    initial = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def bias_variable(shape, name):
    initial = tf.constant_initializer(0.01)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(x) :
    return tf.layers.flatten(x)


def hw_flatten(x) :
    return tf.reshape(x, shape=[-1, x.get_shape()[1] * x.get_shape()[2], x.get_shape()[-1]])


def deconv2d(batch_input, out_channels, kernel_size):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer=initializer)


def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)


def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)
    if batch_nn:
        net = _instance_norm(net)
    return net


def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def global_concat(tensor, concat_layer, batch_size):
    h = tf.shape(tensor)[1]
    w = tf.shape(tensor)[2]
    concat_t = tf.squeeze(concat_layer, [1, 2])
    dims = concat_t.get_shape()[-1]
    batch_l = tf.unstack(concat_t, num=batch_size, axis=0)
    bs = []
    for batch in batch_l:
        batch = tf.tile(batch, [h * w])
        batch = tf.reshape(batch, [h, w, -1])
        bs.append(batch)
    concat_t = tf.stack(bs)
    concat_t.set_shape(concat_t.get_shape().as_list()[:3] + [dims])
    tensor = tf.concat([tensor, concat_t], axis = 3)
    return tensor


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
