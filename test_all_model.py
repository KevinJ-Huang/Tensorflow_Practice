import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc
from PIL import Image
import time
import logging
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/userhome/Enhance')

from net import unet
SCALE = 1



logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


input=tf.placeholder(tf.float32,shape=[None,None,None,3])
with tf.variable_scope('inference'):
    output=unet(input)

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


input_path = "/userhome/dped/validation/input/"
target_path = "/userhome/dped/validation/output/"

chkpt_path = tf.train.get_checkpoint_state("/userhome/Enhance/checkpoint/")
test_images = os.listdir(target_path)
num_test_images = len(test_images)


sess=tf.Session()
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU

step = []
index = []

if chkpt_path:
  for i in range(len(chkpt_path.all_model_checkpoint_paths)):
      print("loaded"+chkpt_path.all_model_checkpoint_paths[i])
      saver.restore(sess,chkpt_path.all_model_checkpoint_paths[i])
      psnr_total = 0
      for j in range(num_test_images):
          image_phone = misc.imread(input_path + test_images[j])/255.0
          image_dslr = misc.imread(target_path + test_images[j])
          image = np.reshape(image_phone, [1, image_phone.shape[0], image_phone.shape[1], 3])
          enhanced = sess.run(output, feed_dict={input: image})
          enhanced = np.clip(enhanced, 0, 1)
          enhanced = np.reshape(enhanced, [image_phone.shape[0], image_phone.shape[1], 3])*255.0
          psnr_instance = output_psnr_mse(enhanced/255.0, image_dslr/255.0)
          psnr_total = psnr_total + psnr_instance
      psnr_p = psnr_total / num_test_images

      step.append(int(chkpt_path.all_model_checkpoint_paths[i].split('-')[-1]))
      index.append(psnr_p)

      log.info("  Evaluation average PSNR = {:.2f} dB".format(psnr_p))


  plt.plot(np.array(step),np.array(index))
  plt.xlabel('step')
  plt.ylabel('psnr')
  plt.savefig('/userhome/Enhance/checkpoint/val_psnr.jpg')


