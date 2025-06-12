import os
import tensorflow as tf
from losses.loss_functions import discriminator_loss, generator_loss


def gradient_penalty(discriminator_fn, real_samples, fake_samples):
    batch_size = tf.shape(real_samples)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        output = discriminator_fn(interpolated, training=True)
        output = tf.reduce_mean(output, axis=[1, 2, 3])
    fake = tf.ones_like(output)
    gradients = tape.gradient(output, interpolated, output_gradients=fake)
    gradients = tf.reshape(gradients, [batch_size, -1])
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
    penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
    return penalty

def make_train_step_fn(generator, discriminator, gen_optimizer, disc_optimizer,
                       lambda_gp=50, lambda_ext=30, z_dim=32, thr=20.0):

    def train_step(coarse_batch, fine_batch):
        batch_size = tf.shape(coarse_batch)[0]

        def sample_noise():
            return tf.random.normal(shape=(batch_size, z_dim))

        with tf.GradientTape() as disc_tape:
            z = sample_noise()
            fake_images = generator([coarse_batch, z], training=True)
            real_out, real_extreme = discriminator(fine_batch, training=True)
            fake_out, fake_extreme = discriminator(fake_images, training=True)

            label_real = tf.cast(tf.reduce_max(fine_batch, axis=[1, 2, 3]) > thr, tf.float32)
            label_fake = tf.cast(tf.reduce_max(fake_images, axis=[1, 2, 3]) > thr, tf.float32)

            gp = gradient_penalty(lambda x, training=False: discriminator(x, training=training)[0], fine_batch, fake_images)

            d_loss = discriminator_loss(
                real_output=tf.reduce_mean(real_out, axis=[1, 2, 3]),
                fake_output=tf.reduce_mean(fake_out, axis=[1, 2, 3]),
                real_extreme=real_extreme, fake_extreme=fake_extreme,
                label_real=label_real, label_fake=label_fake,
                gp=gp, lambda_gp=lambda_gp, lambda_ext=lambda_ext)

        disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            z = sample_noise()
            fake_images = generator([coarse_batch, z], training=True)
            fake_out, fake_extreme = discriminator(fake_images, training=True)
            g_loss = generator_loss(
                fake_out=tf.reduce_mean(fake_out, axis=[1, 2, 3]),
                fake_img=fake_images,
                real_img=fine_batch,
                pred_extreme=fake_extreme, lambda_ext=lambda_ext)

        gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        return d_loss, g_loss

    return train_step

def train_wgan_gp(coarse_batches, fine_batches,
                  generator, discriminator,
                  gen_optimizer, disc_optimizer,
                  strategy,
                  total_iterations=200,
                  n_critic=5,
                  print_interval=10,
                  save_interval=50,
                  save_dir='./checkpoints',
                  lambda_gp=1e-3,
                  lambda_ext=30,
                  z_dim=32):

    train_step_fn = make_train_step_fn(generator, discriminator, gen_optimizer, disc_optimizer,
                                       lambda_gp=lambda_gp, lambda_ext=lambda_ext, z_dim=z_dim)

    def distributed_train_step(coarse_batch, fine_batch):
        return strategy.run(train_step_fn, args=(coarse_batch, fine_batch))

    os.makedirs(save_dir, exist_ok=True)
    d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
    g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
    iteration = 0

    while iteration < total_iterations:
        iteration += 1
        for _ in range(n_critic):
            batch_idx = iteration % len(coarse_batches)
            d_loss_replica, _ = distributed_train_step(coarse_batches[batch_idx], fine_batches[batch_idx])
            d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_replica, axis=None)
            d_loss_metric.update_state(d_loss)

        batch_idx = iteration % len(coarse_batches)
        _, g_loss_replica = distributed_train_step(coarse_batches[batch_idx], fine_batches[batch_idx])
        g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_replica, axis=None)
        g_loss_metric.update_state(g_loss)

        if iteration % print_interval == 0:
            print(f"Epoch {iteration}, D Loss: {d_loss_metric.result().numpy():.4f}, G Loss: {g_loss_metric.result().numpy():.4f}")
            d_loss_metric.reset_state()
            g_loss_metric.reset_state()

        if iteration % save_interval == 0:
            generator.save(os.path.join(save_dir, f'generator_iter_{iteration}.keras'))
            discriminator.save(os.path.join(save_dir, f'discriminator_iter_{iteration}.keras'))