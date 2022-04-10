import scipy.stats as st
import tensorflow as tf
import numpy as np
import sys
import cv2

from functools import reduce

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def get_batch_attention(input_image, batch_size, patch_size):
    input_image = np.reshape(input_image, (-1, patch_size, patch_size, 3))
    out = np.zeros(shape=[batch_size, patch_size, patch_size])
    for i in range(batch_size):
        image = input_image[i, :, :, :]
        image = np.reshape(image, (patch_size, patch_size, 3))
        image = (image * 255).astype(dtype='uint8')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        attention = 1 - norm
        blurred = cv2.GaussianBlur(attention, (101, 101), 101)
        blurred = np.reshape(blurred, (1, patch_size, patch_size))
        out[i, :, :] = blurred
    return out


def contrast(input):
    input_gray = tf.image.rgb_to_grayscale(input)
    batch_size, width, height, channel = input_gray.shape
    input_gray = tf.reshape(input_gray, shape=[-1, width, height])
    input_gray = tf.pad(input_gray, [[0, 0], [1, 1], [1, 1]], mode="SYMMETRIC")
    contrast_filter_1 = tf.constant([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=tf.float32)  # m - l
    contrast_filter_2 = tf.constant([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=tf.float32)  # m - t
    contrast_filter_3 = tf.constant([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=tf.float32)  # m - b
    contrast_filter_4 = tf.constant([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=tf.float32)  # m - r
    input_gray = tf.cast(input_gray, dtype=tf.float32)
    width1 = width + 2
    height1 = height + 2
    input_gray = tf.reshape(input_gray, shape=[-1, width1, height1, 1])
    contrast_filter_1 = tf.reshape(contrast_filter_1, shape=[3, 3, 1, 1])
    contrast_filter_2 = tf.reshape(contrast_filter_2, shape=[3, 3, 1, 1])
    contrast_filter_3 = tf.reshape(contrast_filter_3, shape=[3, 3, 1, 1])
    contrast_filter_4 = tf.reshape(contrast_filter_4, shape=[3, 3, 1, 1])
    m_l = tf.nn.conv2d(input_gray, contrast_filter_1, strides=[1, 1, 1, 1], padding="VALID", name="m_l")
    m_t = tf.nn.conv2d(input_gray, contrast_filter_2, strides=[1, 1, 1, 1], padding="VALID", name="m_t")
    m_b = tf.nn.conv2d(input_gray, contrast_filter_3, strides=[1, 1, 1, 1], padding="VALID", name="m_b")
    m_r = tf.nn.conv2d(input_gray, contrast_filter_4, strides=[1, 1, 1, 1], padding="VALID", name="m_r")
    x = tf.reduce_sum((m_t*m_t)) + tf.reduce_sum((m_l*m_l)) + tf.reduce_sum((m_b*m_b)) + tf.reduce_sum((m_r*m_r))
    d = width * height * 4 - (width * 2 + height * 2)
    d = tf.cast(d, dtype=tf.float32)
    return x / d


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter


def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')


def sobel(input):
    input_gray = tf.image.rgb_to_grayscale(input)
    bs, w, h, c = input_gray.shape
    # input_gray = tf.reshape(input_gray, shape=[-1, w, h])
    Gx = np.float32(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    Gy = np.float32(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    Gx = tf.expand_dims(Gx, axis=2)
    Gx = tf.expand_dims(Gx, axis=3)
    Gy = tf.expand_dims(Gy, axis=2)
    Gy = tf.expand_dims(Gy, axis=3)
    x = tf.nn.conv2d(input_gray, Gx, strides=[1, 1, 1, 1], padding="SAME", name="hor")
    y = tf.nn.conv2d(input_gray, Gy, strides=[1, 1, 1, 1], padding="SAME", name="vec")
    G = tf.sqrt(tf.square(x) + tf.square(y))
    return G



def process_command_args(arguments):

    # specifying default parameters

    batch_size = 32
    train_size = 4000
    learning_rate = 1e-4
    num_train_iters = 40000

    w_content = 100
    w_psnr = 1000
    w_gan = 1
    w_color = 0
    w_contrast = 100

    dped_dir = '../dped2/dped/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    eval_step = 1000

    phone = ""

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("w_content"):
            w_content = float(args.split("=")[1])

        if args.startswith("w_psnr"):
            w_psnr = float(args.split("=")[1])

        if args.startswith("w_gan"):
            w_gan = float(args.split("=")[1])

        # -----------------------------------

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])


    if phone == "":
        print("\nPlease specify the camera model by running the script with the following parameter:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    if phone not in ["iphone", "sony", "blackberry", "FiveK_128", "FiveK_256"]:
        print("\nPlease specify the correct camera model:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    print("\nThe following parameters will be applied for training:\n")

    print("Phone model:", phone)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Training size:", train_size)
    print("Training iterations:", str(num_train_iters))
    print()
    print("Content loss:", w_content)
    print("Psnr loss:", w_psnr)
    print("Gan loss:", w_gan)
    print("Contrast loss:", w_contrast)
    print()
    print("Path to DPED dataset:", dped_dir)
    print("Path to VGG-19 network:", vgg_dir)
    print("Evaluation step:", str(eval_step))
    print()
    return phone, batch_size, train_size, learning_rate, num_train_iters, \
            w_content, w_psnr, w_gan, w_color, w_contrast, \
           dped_dir, vgg_dir, eval_step


def process_test_model_args(arguments):

    phone = ""
    dped_dir = '../dped2/dped/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the model by running the script with the following parameter:\n")
        print("python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu

def get_resolutions():

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["FiveK_128"] = [1024, 1536]
    res_sizes["FiveK_256"] = [344, 512]
    res_sizes["iphone_orig"] = [1536, 2048]
    # res_sizes["blackberry"] = [3120, 3120]
    res_sizes["blackberry"] = [1560, 2080]
    # res_sizes["blackberry"] = [384, 512]
    # res_sizes["blackberry"] = [768, 1024]
    # res_sizes["blackberry_orig"] = [3120, 4160]
    res_sizes["blackberry_orig"] = [1560, 2080]
    # res_sizes["blackberry_orig"] = [1536, 2048]
    res_sizes["sony"] = [1944, 2592]
    # res_sizes["sony"] = [1024, 1280]
    res_sizes["sony_orig"] = [1944, 2592]
    # res_sizes["sony_orig"] = [1920, 2560]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]

    return res_sizes

def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


def extract_crop(image, resolution, phone, res_sizes):

    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up : y_down, x_up : x_down, :]
