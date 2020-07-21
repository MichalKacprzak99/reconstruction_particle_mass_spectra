from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import uproot
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

file = uproot.open("files\PhaseSpaceSimulation.root")
events = file["PhaseSpaceTree"]
NumEntries = events.numentries  #number of data objects (vectors)
params = ["H1_PX","H1_PY","H1_PZ","H2_PX","H2_PY","H2_PZ","H3_PX","H3_PY","H3_PZ"]

Mass_K = 493.677

class GAN:
    def __init__(self):

        self.vec_shape = (9,)

        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)

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
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(66, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(142))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.vec_shape), activation='tanh'))
        model.add(Reshape(self.vec_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        vec = model(noise)

        return Model(noise, vec)

    def build_discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.vec_shape))
        model.add(Dense(40))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(80))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        vec = Input(shape=self.vec_shape)
        validity = model(vec)

        return Model(vec, validity)
    #do przystosowania po załadowaniu danych
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        data = events.lazyarrays(params)
        data_arr = np.vstack(list(data[elem] for elem in params)).T

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18,8)
        Mass_B = np.zeros(NumEntries)
        P1x = np.zeros(NumEntries)
        for i in range(NumEntries):
            p_products = np.array([np.sqrt(np.square(data_arr[i][0]) + np.square(data_arr[i][1]) + np.square(data_arr[i][2])),
                       np.sqrt(np.square(data_arr[i][3]) + np.square(data_arr[i][4]) + np.square(data_arr[i][5])),
                       np.sqrt(np.square(data_arr[i][6]) + np.square(data_arr[i][7]) + np.square(data_arr[i][8]))])
            p_total = np.sqrt(np.square(data_arr[i][0]+data_arr[i][3]+data_arr[i][6]) +
                       np.square(data_arr[i][1]+data_arr[i][4]+data_arr[i][7]) +
                       np.square(data_arr[i][2]+data_arr[i][5]+data_arr[i][8]))
            E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
            Mass_B[i] = math.sqrt(np.sum(E_total) ** 2  - p_total ** 2)
            P1x[i] = data_arr[i][0]


            #print(E_total)
        n, bins, patches = ax1.hist(Mass_B, 100, range =(5279,5279.3))
        ax1.set_xlabel('Mass of the B meson [MeV]')
        ax1.set_ylabel('Number of counts')
        n2, bins2, patches2 = ax2.hist(P1x, 100, range = (-100000,100000))
        ax2.set_xlabel('Momentum of the K1 [MeV]')
        ax2.set_ylabel('Number of counts')
        fig.savefig("images/InputData.png")
        plt.close()


        # Rescale -1 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(data_arr)
        scaled_data = scaler.transform(data_arr)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        Average_mass_predicted = []
        MPV_mass_predicted = []

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

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, scaler, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x, Mass_B)

    def sample_images(self, epoch, scaler, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x_data, Mass_B_data):

        r = 10000;
        noise = np.random.normal(0, 1, (r , self.latent_dim))
        gen_raw = self.generator.predict(noise)
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #scaler = scaler.fit(gen_raw)
        gen_p = scaler.inverse_transform(gen_raw)
        #print(gen_p)
        cnt = 0;
        fig, axs = plt.subplots(2,2)
        fig.set_size_inches(18,11)
        Mass_B = np.zeros(r)
        P1x = np.zeros(r)
        for i in range(r):
            p_products = np.array([np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1]) + np.square(gen_p[i][2])),
                       np.sqrt(np.square(gen_p[i][3]) + np.square(gen_p[i][4]) + np.square(gen_p[i][5])),
                       np.sqrt(np.square(gen_p[i][6]) + np.square(gen_p[i][7]) + np.square(gen_p[i][8]))])
            p_total = np.sqrt(np.square(gen_p[i][0]+gen_p[i][3]+gen_p[i][6]) +
                       np.square(gen_p[i][1]+gen_p[i][4]+gen_p[i][7]) +
                       np.square(gen_p[i][2]+gen_p[i][5]+gen_p[i][8]))
            E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
            Mass_B[i] = math.sqrt(np.sum(E_total) ** 2  - p_total ** 2)
            P1x[i] = gen_p[i][0]
                #axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                #axs[i,j].axis('off')

        Average_mass_predicted.append(np.mean(Mass_B))

        n, bins, patches = axs[0,0].hist(Mass_B, 200, range =(0,200000), alpha = 0.5, label = 'Generated data')
        #axs[0,0].hist(Mass_B_data[0:10000], 200, range=(0,50000), alpha=0.5, label = 'Input data')
        axs[0,0].set_xlabel('Mass of the B meson [MeV]')
        axs[0,0].set_ylabel('Number of counts')
        axs[0,0].axvline(x=np.mean(Mass_B_data), color ='r', linestyle = 'dashed')
        #axs[0,0].legend(loc='upper right')
        MPV_mass_predicted.append(np.mean(bins[np.where(n == np.amax(n))]))

        n2, bins2, patches2 = axs[0,1].hist(P1x, 100, range = (-100000,100000), alpha = 0.5, label = 'Generated data')
        axs[0,1].hist(P1x_data[0:10000], 100, range = (-100000,100000), label = 'Input data', alpha = 0.5)
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_xlabel('Momentum of the K1 [MeV]')
        axs[0,1].set_ylabel('Number of counts')

        axs[1,0].plot(range(0,epoch+sample_interval,sample_interval), Average_mass_predicted, c='r', linewidth=4.0)
        axs[1,0].set_xlim([0,30000])
        axs[1,0].set_ylim([100,500000])
        axs[1,0].set_xlabel('Epoch number')
        axs[1,0].set_ylabel('Mean of the B mass predicted')
        axs[1,0].plot(range(0,30000+sample_interval,200), np.zeros(int(30000/200) +1)+np.mean(Mass_B_data),'m-.')
        axs[1,0].set_yscale('log')


        axs[1, 1].plot(range(0, epoch + sample_interval, sample_interval), MPV_mass_predicted, c='g', linewidth=4.0)
        axs[1, 1].set_xlim([0, 30000])
        axs[1, 1].set_ylim([100, 500000])
        axs[1, 1].set_xlabel('Epoch number')
        axs[1, 1].set_ylabel('MPV of the B mass predicted')
        axs[1, 1].plot(range(0, 30000 + sample_interval, 200), np.zeros(int(30000/200) +1)+np.mean(Mass_B_data), 'm-.')
        axs[1, 1].set_yscale('log')
        #fig.savefig("images/%d.png" % epoch)


        fig.savefig("images/%d.png" % epoch)


        plt.close()


if __name__ == '__main__':


    gan = GAN()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)
