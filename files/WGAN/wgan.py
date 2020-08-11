from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding1D, LeakyReLU, Conv1D, UpSampling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
from settings import create_images_folder, sample_images, prepare_data
import numpy as np


class WGAN():
    def __init__(self):
        self.scaler = None
        self.vec_shape = (10, 1)
        self.latent_dim = 100
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.disctiminator = self.build_discriminator()
        self.disctiminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        # print(z)
        img = self.generator(z)

        # # For the combined model we will only train the generator
        self.disctiminator.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.disctiminator(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(320, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((5, 64)))
        model.add(UpSampling1D())
        model.add(Conv1D(16, kernel_size=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv1D(1, kernel_size=1, padding="same"))
        model.add(Activation("tanh"))

        print("gen")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv1D(1, kernel_size=3, strides=1, input_shape=self.vec_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding1D(padding=(0, 1)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        print("dis")
        model.summary()

        img = Input(shape=self.vec_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        self.scaler, scaled_data, P1x, P3z, P_tot, E_tot, Mass_B = prepare_data()

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        Average_mass_predicted = []
        MPV_mass_predicted = []
        G_loss_epochs = []
        D_loss_epochs = []
        D_acc_epochs = []
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, scaled_data.shape[0], batch_size)
                vecs = scaled_data[idx]
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_vecs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.disctiminator.train_on_batch(vecs, valid)
                d_loss_fake = self.disctiminator.train_on_batch(gen_vecs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                #
                # # Clip critic weights
                for l in self.disctiminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            # print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            G_loss_epochs.append(g_loss)
            D_loss_epochs.append(d_loss[0])
            D_acc_epochs.append(d_loss[1])
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                sample_images(self, epoch, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x, P3z, P_tot,
                              E_tot, Mass_B, G_loss_epochs, D_loss_epochs, D_acc_epochs)
