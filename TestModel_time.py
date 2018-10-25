import tensorflow as tf
import time
from net import unet
import argparse
import numpy as np

def main(args):
    x = tf.placeholder(dtype=tf.float32,shape=[1,args.full_size1,args.full_size2,3],name='input_ori')
    x_low = tf.placeholder(dtype=tf.float32, shape=[1, args.low_size, args.low_size, 3], name='input_low')
    input_ori = np.random.randn(1,args.full_size1,args.full_size2,3)
    input_low = np.random.randn(1, args.low_size, args.low_size, 3)

    # image = tf.random_normal(shape=[1,args.full_size1,args.full_size2,3])
    # image_low = tf.random_normal(shape=[1,args.low_size,args.low_size,3])
    out = unet(x_low)


    config = None
    with tf.Session(config) as sess:
        sess.run(tf.global_variables_initializer())
        time_start = int(round(time.time() * 1000))
        for i in range(args.iters):
            output = sess.run(out,feed_dict={x_low:input_low})
            # output = sess.run(out, feed_dict={ x: input_ori})
            # out = unet(image,image_low)
        time_end = int(round(time.time()*1000))
        print("ms:%.1f ms"%((time_end-time_start)/args.iters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_size1', default=2048, type=float,help='The size of input_ori.')
    parser.add_argument('--full_size2', default=2048, type=float, help='The size of input_ori.')
    parser.add_argument('--low_size', default=512, type=float, help='The size of input')
    parser.add_argument('--iters', default=10, type=float,help='iters for test.')
    args = parser.parse_args()
    main(args)
