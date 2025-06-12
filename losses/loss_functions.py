import tensorflow as tf

def ssim_loss(fake_images, real_images):
    ssim_index = tf.image.ssim(fake_images, real_images, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim_index)

def weighted_mae(fake, real, thr=20.0, tail_w=5.0):
    weights = 1. + (tail_w - 1.) * tf.cast(real > thr, tf.float32)
    return tf.reduce_mean(weights * tf.abs(fake - real))

def exloss(fake, real, alpha=2.0, thr=20.0):
    diff = real - fake
    scale = 1. + alpha * tf.nn.relu(diff) / tf.maximum(real, thr)
    return tf.reduce_mean(scale * tf.square(diff))

def quantile_loss(pred, target, q=0.95):
    e = target - pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

def discriminator_loss(real_output, fake_output, 
                       real_extreme, fake_extreme, 
                       label_real, label_fake, 
                       gp, lambda_gp=10.0, lambda_ext=10.0):
    w_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_ext_real = bce(label_real, real_extreme)
    loss_ext_fake = bce(label_fake, fake_extreme)
    ext_loss = loss_ext_real + loss_ext_fake
    return w_loss + lambda_gp * gp + lambda_ext * ext_loss

def generator_loss(fake_out, fake_img, real_img, pred_extreme,
                   penalty_weight=1.0, heavy_weight=50.0,
                   l1_weight=5.0, ssim_weight=100.0,
                   quantile=0.90, thr=20.0,
                   lambda_ext=50.0):

    w_loss = -tf.reduce_mean(fake_out)
    ssim_term = ssim_weight * ssim_loss(fake_img, real_img)
    l1_term = l1_weight * tf.reduce_mean(tf.abs(fake_img - real_img))
    quant_term = penalty_weight * quantile_loss(fake_img, real_img, q=quantile)

    mask_heavy = tf.cast(real_img >= thr, tf.float32)
    heavy_mse = tf.reduce_mean(mask_heavy * tf.square(fake_img - real_img))
    heavy_term = heavy_weight * heavy_mse

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    extreme_term = lambda_ext * bce(tf.ones_like(pred_extreme), pred_extreme)

    return w_loss + extreme_term + ssim_term
