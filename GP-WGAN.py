import tensorflow as tf



def texture_loss(target, prediction, adv_):
  enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(prediction), [-1, PATCH_WIDTH * PATCH_HEIGHT])
  dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(target), [-1, PATCH_WIDTH * PATCH_HEIGHT])
  adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
  adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

  discrim_predictions = adversarial(adversarial_image)


  gradients = tf.gradients(discrim_predictions, adversarial_image)[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
  gp = tf.reduce_mean((slopes - 1.) ** 2)

  discrim_target = tf.concat([adv_, 1 - adv_], 1)

  loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))+gp*10.0
  loss_texture = tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))

  correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
  discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  return loss_texture, discim_accuracy,loss_discrim