import os
import tensorflow as tf
import argparse

class DataPipeLine(object):
    def __init__(self,path):
        self.path = path
    def produce_one_samples(self):
        dirname = os.path.dirname(self.path)
        with open(self.path,'r') as fid:
            flist = [l.strip() for l in fid.xreadlines()]
        input_files = [os.path.join(dirname, 'iphone', f) for f in flist]
        output_files = [os.path.join(dirname,'canon',f) for f in flist]
        input_queue,output_queue = tf.train.slice_input_producer([input_files,output_files],shuffle=True,
                                                                 seed=1234, num_epochs=None)
        input_file = tf.read_file(input_queue)
        output_file = tf.read_file(output_queue)
        im_input = tf.image.decode_jpeg(input_file,channels=3)
        im_output = tf.image.decode_jpeg(output_file,channels=3)
        sample = {}
        with tf.name_scope('normalize_images'):
            im_input = tf.to_float(im_input) / 255.0
            im_output = tf.to_float(im_output) / 255.0
        inout = tf.concat([im_input,im_output],axis=2)
        inout.set_shape([None, None, 6])
        inout = tf.image.resize_images(inout,[100,100])

        sample['input'] = inout[:, :, :3]
        sample['output'] = inout[:, :, 3:]
        return sample


def main(args):
    sample = DataPipeLine(args.data_dir).produce_one_samples()
    samples = tf.train.batch(sample,batch_size=args.batch_size,
                                     num_threads=2,
                                     capacity=32)
    loss = tf.reduce_sum(tf.pow(samples['input'] - samples['output'], 2)) / (2 * args.batch_size)

    total_batch = int(400 / args.batch_size)
    sv = tf.train.Supervisor()
    total_loss = 0
    with sv.managed_session() as sess:
        step = 0
        while True:
            if sv.should_stop():
                print("stopping supervisor")
                break
            try:
                loss_ = sess.run( loss)
                total_loss += loss_
                step += 1
                print("step:%d,loss:%.2f" %(step,loss_))
                if step%total_batch == 0:
                    print("%d epochs,total loss:%.2f" %((step/total_batch),total_loss))
                    total_loss = 0
            except tf.errors.AbortedError:
                print("Aborted")
                break
            except KeyboardInterrupt:
                break
        sv.request_stop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='test_data/dataset.txt',help="The path of input txt")
    parser.add_argument("--batch_size", default=40, help="Number Images of each batch")
    parser.add_argument("--epochs", default=30, help="The number of epochs")
    args = parser.parse_args()
    main(args)