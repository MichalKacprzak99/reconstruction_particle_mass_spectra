import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
tf.compat.v1.disable_eager_execution()
# Sample z from uniform distribution
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

# Input image, for discriminator model.
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])

# Input noise for generator.
Z = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])

def generator(z):
    with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
        x = tf.compat.v1.layers.dense(z, 128, activation=tf.nn.relu)
        x = tf.compat.v1.layers.dense(z, 784)
        x = tf.nn.sigmoid(x)
    return x

def discriminator(x):
    with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
        x = tf.compat.v1.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.compat.v1.layers.dense(x, 1)
        x = tf.nn.sigmoid(x)
    return x

# Generator model
G_sample = generator(Z)

# Discriminator models
D_real = discriminator(X)
D_fake = discriminator(G_sample)


# Loss function
D_loss = -tf.reduce_mean(input_tensor=tf.math.log(D_real) + tf.math.log(1. - D_fake))
G_loss = -tf.reduce_mean(input_tensor=tf.math.log(D_fake))

# Select parameters
disc_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith("disc")]
gen_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith("gen")]

# Optimizers
D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=disc_vars)
G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=gen_vars)

# Batch size
mb_size = 128

# Dimension of input noise
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

if not os.path.exists('out2/'):
    os.makedirs('out2/')

i = 0

for it in range(1000000):

    # Save generated images every 1000 iterations.
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)


    # Get next batch of images. Each batch has mb_size samples.
    X_mb, _ = mnist.train.next_batch(mb_size)


    # Run disciminator solver
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    # Run generator solver
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    # Print loss
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))