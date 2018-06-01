import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)

N = 200
x_data = np.linspace(-1, 1, N)
y_data = 2.0*x_data + np.random.standard_normal(x_data.shape)*0.3 + 0.5
x_data = x_data.reshape([N, 1])
y_data = y_data.reshape([N, 1])


def log_hook(sess, log_fetches):
    data = sess.run(log_fetches)
    step = data['step']
    loss = data['loss']
    log.info('Step{} | loss = {:.4f}'.format(step, loss))

def main(args):
    x = tf.to_float(x_data)
    y = tf.to_float(y_data)
    W = tf.Variable(tf.random_normal([1,1],stddev=0.1))
    b = tf.Variable(tf.random_normal([1],stddev=0.1))
    pred = tf.matmul(x, W) + b
    loss = tf.reduce_sum(tf.pow(pred - y,2))

    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss,global_step=global_step)
    tf.summary.scalar('loss',loss)

    log_fetches ={
        "W":W,
        "b":b,
        "loss":loss,
        "step":global_step}


    sv = tf.train.Supervisor(logdir = args.checkpoint_dir,save_summaries_secs=args.summary_interval,
                             save_model_secs=args.checkpoint_interval)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with sv.managed_session(config=config) as sess:
        sv.loop(args.log_interval,log_hook,(sess,log_fetches))
        while True:
            if sv.should_stop():
                log.info('stopping supervisor')
                break
            try:
                WArr, bArr, _ = sess.run([W,b,train_op])
                plt.scatter(x_data, y_data)
                plt.scatter(x_data, WArr*x_data+bArr)
                plt.pause(0.3)
                plt.cla()
            except tf.errors.AbortedError:
                log.error('Aborted')
                break
            except KeyboardInterrupt:
                break
        chkpt_path = os.path.join(args.checkpoint_dir, 'on_stop.ckpt')
        log.info("Training complete, saving chkpt {}".format(chkpt_path))
        sv.saver.save(sess, chkpt_path)
        sv.request_stop()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate for the stochastic gradient update.')
    parser.add_argument('--checkpoint_dir', default='summary/', help='directory of summary to save.')
    parser.add_argument('--summary_interval', type=int, default=1, help='interval between tensorboard summaries (in s)')
    parser.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
    parser.add_argument('--checkpoint_interval', type=int, default=20,help='interval between model checkpoints (in s)')

    args = parser.parse_args()
    main(args)



