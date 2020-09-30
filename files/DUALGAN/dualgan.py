from __future__ import print_function, division
import scipy
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from settings import create_images_folder, prepare_data, Mass_K
import math
import matplotlib.pyplot as plt
import numpy as np


class DUALGAN():
    def __init__(self):
        self.path = create_images_folder(self)
        self.vec_shape = 10

        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.D_A = self.build_discriminator()
        self.D_A.compile(loss=self.wasserstein_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.D_B = self.build_discriminator()
        self.D_B.compile(loss=self.wasserstein_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.G_AB = self.build_generator()
        self.G_BA = self.build_generator()

        # For the combined model we will only train the generators
        self.D_A.trainable = False
        self.D_B.trainable = False

        # The generator takes images from their respective domains as inputs
        imgs_A = Input(shape=(self.vec_shape,))
        imgs_B = Input(shape=(self.vec_shape,))

        # Generators translates the images to the opposite domain
        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        # The discriminators determines validity of translated images
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        # Generators translate the images back to their original domain
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        # The combined model  (stacked generators and discriminators)
        self.combined = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                              optimizer=optimizer,
                              loss_weights=[1, 1, 100, 100])

    def build_generator(self):

        X = Input(shape=(self.vec_shape,))

        model = Sequential()
        model.add(Dense(96, input_dim=self.vec_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))

        model.add(Dense(192))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))


        model.add(Dense(self.vec_shape, activation='tanh'))
        print("gen")
        model.summary()
        X_translated = model(X)

        return Model(X, X_translated)

    def build_discriminator(self):

        img = Input(shape=(self.vec_shape,))

        model = Sequential()
        model.add(Dense(96, input_dim=self.vec_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(192))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1))
        print("dis")
        model.summary()
        validity = model(img)

        return Model(img, validity)

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (scaled_data, _), (_, _) = mnist.load_data()
        self.scaler, scaled_data, P1x, P3z, P_tot, E_tot, Mass_B = prepare_data()
        # Rescale -1 to 1
        scaled_data = scaled_data[..., np.newaxis]
        # Domain A and B (rotated)
        X_A = scaled_data[:int(scaled_data.shape[0] / 2)]
        X_B = scipy.ndimage.interpolation.rotate(scaled_data[int(scaled_data.shape[0] / 2):], 90, axes=(1, 2))

        X_A = X_A.reshape(X_A.shape[0], self.vec_shape)
        X_B = X_B.reshape(X_B.shape[0], self.vec_shape)

        clip_value = 0.01
        n_critic = 4

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        Average_mass_predicted = []
        MPV_mass_predicted = []
        G_loss_epochs = []
        D_A_loss_epochs = []
        D_B_loss_epochs = []

        for epoch in range(epochs):

            # Train the discriminator for n_critic iterations
            for _ in range(n_critic):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Sample generator inputs
                imgs_A = self.sample_generator_input(X_A, batch_size)
                imgs_B = self.sample_generator_input(X_B, batch_size)

                # Translate images to their opposite domain
                fake_B = self.G_AB.predict(imgs_A)
                fake_A = self.G_BA.predict(imgs_B)

                # Train the discriminators
                D_A_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                D_A_loss_fake = self.D_A.train_on_batch(fake_A, fake)

                D_B_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                D_B_loss_fake = self.D_B.train_on_batch(fake_B, fake)

                D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
                D_B_loss = 0.5 * np.add(D_B_loss_real, D_B_loss_fake)

                # Clip discriminator weights
                for d in [self.D_A, self.D_B]:
                    for l in d.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

            # ------------------
            #  Train Generators
            # ------------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

            # Plot the progress
            print("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                  % (epoch, D_A_loss[0], D_B_loss[0], g_loss[0]))
            G_loss_epochs.append(g_loss[0])
            D_A_loss_epochs.append(D_A_loss[0])
            D_B_loss_epochs.append(D_B_loss[1])
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_A, X_B, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x,
                                   P3z, P_tot, E_tot, Mass_B, G_loss_epochs, D_A_loss_epochs, D_B_loss_epochs)

    def sample_images(self, epoch, X_A, X_B, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x_data,
                      P3z_data, P_tot_data, E_tot_data, Mass_B_data, G_loss_epochs, D_loss_epochs, D_acc_epochs):
        r, c = 5000, 5000

        # Sample generator inputs
        imgs_A = self.sample_generator_input(X_A, c)
        imgs_B = self.sample_generator_input(X_B, c)

        # Images translated to their opposite domain
        fake_B = self.G_AB.predict(imgs_A)
        fake_A = self.G_BA.predict(imgs_B)
        gen_p = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])
        gen_p = self.scaler.inverse_transform(gen_p)
        fig, axs = plt.subplots(3, 3)
        fig.set_size_inches(14, 14)
        Mass_B = np.zeros(r)
        P1x = np.zeros(r)
        P3z = np.zeros(r)
        P_tot = np.zeros(r)
        E_tot = np.zeros(r)
        for i in range(r):
            p_products = np.array([np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1]) + np.square(gen_p[i][2])),
                                   np.sqrt(np.square(gen_p[i][3]) + np.square(gen_p[i][4]) + np.square(gen_p[i][5])),
                                   np.sqrt(np.square(gen_p[i][6]) + np.square(gen_p[i][7]) + np.square(gen_p[i][8]))])
            p_total = np.sqrt(np.square(gen_p[i][0] + gen_p[i][3] + gen_p[i][6]) +
                              np.square(gen_p[i][1] + gen_p[i][4] + gen_p[i][7]) +
                              np.square(gen_p[i][2] + gen_p[i][5] + gen_p[i][8]))
            E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
            Mass_B[i] = math.sqrt(np.sum(E_total) ** 2 - p_total ** 2)
            P1x[i] = gen_p[i][0]
            P3z[i] = gen_p[i][8]
            P_tot[i] = p_total
            E_tot[i] = np.sum(E_total)

        Average_mass_predicted.append(np.mean(Mass_B))

        n, bins, patches = axs[0, 0].hist(Mass_B, 200, range=(5278, 5280), alpha=0.5, label='Generated data')
        # axs[0,0].hist(Mass_B_data[0:10000], 200, range=(0,50000), alpha=0.5, label = 'Input data')
        axs[0, 0].set_xlabel('Mass of the B meson [MeV]')
        axs[0, 0].set_ylabel('Number of counts')
        axs[0, 0].axvline(5279.29, color='r', linestyle='dashed')
        # axs[0,0].legend(loc='upper right')
        MPV_mass_predicted.append(np.mean(bins[np.where(n == np.amax(n))]))

        n2, bins2, patches2 = axs[0, 1].hist(P1x, 100, range=(-100000, 100000), alpha=0.5, label='Generated data')
        axs[0, 1].hist(P1x_data[0:r], 100, range=(-100000, 100000), label='Input data', alpha=0.5)
        axs[0, 1].legend(loc='upper right')
        axs[0, 1].set_xlabel('Momentum X of K1 [MeV]')
        axs[0, 1].set_ylabel('Number of counts')

        axs[1, 0].plot(range(0, epoch + sample_interval, sample_interval), Average_mass_predicted, c='r', linewidth=4.0)
        axs[1, 0].set_xlim([0, 100])
        axs[1, 0].set_ylim([5000, 6000])
        axs[1, 0].set_xlabel('Epoch number')
        axs[1, 0].set_ylabel('Mean of the B mass predicted')
        axs[1, 0].plot(range(0, 100 + sample_interval, 20), np.zeros(int(100 / 20) + 1) + np.mean(Mass_B_data),
                       'm-.')
        axs[1, 0].set_yscale('log')

        axs[1, 1].plot(range(0, epoch + sample_interval, sample_interval), MPV_mass_predicted, c='g', linewidth=4.0)
        axs[1, 1].set_xlim([0, 100])
        axs[1, 1].set_ylim([5000, 6000])
        axs[1, 1].set_xlabel('Epoch number')
        axs[1, 1].set_ylabel('MPV of the B mass predicted')
        axs[1, 1].plot(range(0, 100 + sample_interval, 20), np.zeros(int(100 / 20) + 1) + np.mean(Mass_B_data),
                       'm-.')
        axs[1, 1].set_yscale('log')
        # fig.savefig("images/%d.png" % epoch)

        axs[2, 1].hist(P3z, 100, range=(0, 800000), alpha=0.5, label='Generated data')
        axs[2, 1].hist(P3z_data[0:r], 100, range=(0, 800000), label='Input data', alpha=0.5)
        axs[2, 1].legend(loc='upper right')
        axs[2, 1].set_xlabel('Momentum Z of K3 [MeV]')
        axs[2, 1].set_ylabel('Number of counts')

        axs[2, 0].plot(range(0, epoch + 1, 1), G_loss_epochs, c='g', linewidth=1.0, label='G loss')
        axs[2, 0].plot(range(0, epoch + 1, 1), D_loss_epochs, c='r', linewidth=1.0, label='D loss')
        axs[2, 0].plot(range(0, epoch + 1, 1), D_acc_epochs, c='c', linewidth=1.0, label='D accuracy')
        axs[2, 0].set_xlim([0, 30000])
        axs[2, 0].set_ylim([0, 2])
        axs[2, 0].set_xlabel('Epoch number')
        axs[2, 0].set_ylabel('Relative ratio')

        axs[0, 2].hist(P_tot, 100, range=(0, 1000000), alpha=0.5, label='Generated data')
        axs[0, 2].hist(P_tot_data[0:r], 100, range=(0, 1000000), label='Input data', alpha=0.5)
        axs[0, 2].legend(loc='upper right')
        axs[0, 2].set_xlabel('Total momentum [MeV]')
        axs[0, 2].set_ylabel('Number of counts')

        axs[1, 2].hist(E_tot, 100, range=(0, 1000000), alpha=0.5, label='Generated data')
        axs[1, 2].hist(E_tot_data[0:r], 100, range=(0, 1000000), label='Input data', alpha=0.5)
        axs[1, 2].legend(loc='upper right')
        axs[1, 2].set_xlabel('Total energy [MeV]')
        axs[1, 2].set_ylabel('Number of counts')
        fig.savefig(self.path + f'{epoch}.png')

        plt.close()
