from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
from settings import create_images_folder, sample_images, prepare_data


class BGAN:
    def __init__(self):

        self.vec_shape = (10,)

        self.latent_dim = 100
        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates vectors
        z = Input(shape=(self.latent_dim,))
        vec = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(vec)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.boundary_loss, optimizer=optimizer)

    def boundary_loss(self, y_true, y_pred):
        """
        Boundary seeking loss.
        Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
        """
        return 0.5 * K.mean((K.log(y_pred) - K.log(1 - y_pred)) ** 2)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(96, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(192))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.vec_shape), activation='tanh'))
        model.add(Reshape(self.vec_shape))

        noise = Input(shape=(self.latent_dim,))
        vec = model(noise)

        return Model(noise, vec)

    def build_discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.vec_shape))
        model.add(Dense(96))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(192))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        vec = Input(shape=self.vec_shape)
        validity = model(vec)

        return Model(vec, validity)

    # do przystosowania po załadowaniu danych
    def train(self, epochs, batch_size=128, sample_interval=50):
        self.path = create_images_folder(self)

        self.scaler, scaled_data, P1x, P3z, P_tot, E_tot, Mass_B = prepare_data()
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        Average_mass_predicted = []
        MPV_mass_predicted = []
        G_loss_epochs = []
        D_loss_epochs = []
        D_acc_epochs = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of vectors
            idx = np.random.randint(0, scaled_data.shape[0], batch_size)
            vecs = scaled_data[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new vectors
            gen_vecs = self.generator.predict(noise)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(vecs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_vecs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            G_loss_epochs.append(g_loss)
            D_loss_epochs.append(d_loss[0])
            D_acc_epochs.append(d_loss[1])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                sample_images(self, epoch, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x, P3z, P_tot,
                              E_tot, Mass_B, G_loss_epochs, D_loss_epochs, D_acc_epochs)
