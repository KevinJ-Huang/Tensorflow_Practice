import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import logging
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger('train')
log.setLevel(logging.INFO)

data = np.mat([[0.697,0.460,1],
        [0.774,0.376,1],
        [0.634,0.264,1],
        [0.608,0.318,1],
        [0.556,0.215,1],
        [0.403,0.237,1],
        [0.481,0.149,1],
        [0.437,0.211,1],
        [0.666,0.091,0],
        [0.243,0.267,0],
        [0.245,0.057,0],
        [0.343,0.099,0],
        [0.639,0.161,0],
        [0.657,0.198,0],
        [0.360,0.370,0],
        [0.593,0.042,0],
        [0.719,0.103,0]])

def log_hook(sess,log_fetches):
    data = sess.run(log_fetches)
    loss = data['loss']
    step = data['step']
    log.info('Step {}|loss = {:.4f}'.format(step,loss))


def logistic_regression(W,b,x):
    pred = 1/(1+tf.exp(-(tf.matmul(x,W)+b)))
    return pred


def main(args):
    W = tf.Variable(tf.random_normal([2,1],stddev=0.1))
    b = tf.Variable(tf.random_normal([1],stddev=0.1))
    x = tf.to_float(data[:,0:2])
    y = tf.to_float(data[:,2])
    global_step = tf.contrib.framework.get_or_create_global_step()
    pred = logistic_regression(W,b,x)
    loss = tf.reduce_sum(-tf.reshape(y,[-1,1])*tf.log(pred)-(1-tf.reshape(y,[-1,1]))*tf.log(1-pred))
    train_op = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss,global_step)

    tf.summary.scalar('loss',loss)
    log_fetches ={
        "W":W,
        "b":b,
        "loss":loss,
        "step":global_step
    }
    sv = tf.train.Supervisor(logdir = args.checkpoint_dir,save_model_secs=args.checkpoint_interval,
                             save_summaries_secs=args.summary_interval)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with sv.managed_session(config=config) as sess:
        sv.loop(args.log_interval,log_hook,[sess,log_fetches])
        while True:
            if sv.should_stop():
                log.info('stopping supervisor')
            try:
                WArr,bArr,_ = sess.run([W,b,train_op])
                x0 = np.array(data[:8])
                x0_ = np.array(data[8:])
                plt.scatter(x0[:,0],x0[:,1],c='r',label='+')
                plt.scatter(x0_[:,0],x0_[:,1],c='b',label='-')
                x1 = np.arange(-0.2,1.0,0.1)
                y1 = (-bArr-WArr[0]*x1)/WArr[1]
                plt.plot(x1,y1)
                plt.pause(0.01)
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",default=1e-1,type=float,help='learning rate for the stochastic gradient update.')
    parser.add_argument('--checkpoint_dir', default='summary/', help='directory of summary to save.')
    parser.add_argument('--summary_interval', type=int, default=1, help='interval between tensorboard summaries (in s)')
    parser.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='interval between model checkpoints (in s)')

    args =  parser.parse_args()
    main(args)
