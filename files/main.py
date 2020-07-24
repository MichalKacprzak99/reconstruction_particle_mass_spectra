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
import tensorflow as tf

file = uproot.open("PhaseSpaceSimulation.root")
events = file["PhaseSpaceTree"]
NumEntries = events.numentries  #number of data objects (vectors)
params = ["H1_PX","H1_PY","H1_PZ","H2_PX","H2_PY","H2_PZ","H3_PX","H3_PY","H3_PZ"]

Mass_K = 493.677

class GAN:
    def __init__(self):

        self.vec_shape_G = (10,)
        self.vec_shape_D = (10,)

        self.latent_dim = 10
        optimizer = Adam(0.0003, 0.5)

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
        #vec = tf.Tensor("model_1/Identity:0", shape=(None, 9), dtype = np.float32)
        validity = self.discriminator(vec)
        print(validity)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(25, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(50))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.vec_shape_G), activation='tanh'))
        model.add(Reshape(self.vec_shape_G))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        vec = model(noise)

        return Model(noise, vec)

    def build_discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.vec_shape_D))
        model.add(Dense(50))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(25))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        vec = Input(shape=self.vec_shape_D)
        validity = model(vec)

        return Model(vec, validity)
    #do przystosowania po załadowaniu danych
    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        data = events.lazyarrays(params)
        #data["H1_PZ"] = data["H1_PZ"]/4
        #data["H2_PZ"] = data["H2_PZ"] / 4
        #data["H3_PZ"] = data["H3_PZ"] / 4
        data_arr = np.vstack(list(data[elem] for elem in params)).T

        Mass_B = np.zeros(NumEntries)
        P1x = np.zeros(NumEntries); P3z = np.zeros(NumEntries); P1_tot = np.zeros(NumEntries); E_tot = np.zeros(NumEntries) ; Px_tot = np.zeros(NumEntries);
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
            P3z[i] = p_total
            P1_tot[i] = np.sqrt(np.square(data_arr[i][0]) + np.square(data_arr[i][1]) + np.square(data_arr[i][2]))
            Px_tot[i] = data_arr[i][0]+data_arr[i][3]+data_arr[i][6]
            E_tot[i] = np.sum(E_total)

        # Rescale -1 to 1


        #dummy_list = list();
        #for i in range(NumEntries):
        #    dummy_list.append(5000)
        data_arr = np.hstack((data_arr, Mass_B[:, None]))
        #data_arr[1][9] = data_arr[1][9] + 4999000
        #data_arr[2][9] = data_arr[2][9] - 5000000
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(data_arr)

        #data_arr[1][9] = Mass_B[1]
        #data_arr[2][9] = Mass_B[2]

        scaled_data = scaler.transform(data_arr)

        for i in range(NumEntries):
            #print(scaled_data[i][9])
            scaled_data[i][9] = 2*math.atan(data_arr[i][9])/math.pi


        #print(scaled_data)
        #for i in range(NumEntries):
        #    del scaled_data[i][9]
        # Adversarial ground truths
        #scaled_data = np.delete(scaled_data, 9, axis = 1)
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
            #gen_vecs = np.c_[gen_vecs, np.zeros(batch_size)]
            gen_p = scaler.inverse_transform(gen_vecs)
            for i in range(batch_size):
                dmp_products = np.array(
                    [np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1]) + np.square(gen_p[i][2])),
                     np.sqrt(np.square(gen_p[i][3]) + np.square(gen_p[i][4]) + np.square(gen_p[i][5])),
                     np.sqrt(np.square(gen_p[i][6]) + np.square(gen_p[i][7]) + np.square(gen_p[i][8]))])
                dmp_total = np.sqrt(np.square(gen_p[i][0] + gen_p[i][3] + gen_p[i][6]) +
                                  np.square(gen_p[i][1] + gen_p[i][4] + gen_p[i][7]) +
                                  np.square(gen_p[i][2] + gen_p[i][5] + gen_p[i][8]))
                dmE_total = np.sqrt(np.square(dmp_products) + Mass_K ** 2)
                dmMass_B = math.sqrt(np.sum(dmE_total) ** 2 - dmp_total ** 2)
                gen_vecs[i][9] = 2*math.atan(dmMass_B)/math.pi
                #gen_p[i][9] = 5000

            #gen_vecs = scaler.transform(gen_p)
            #for i in range(batch_size):
            #    gen_vecs[i][9] = 2*math.atan(gen_[i][9])/math.pi
            #print(gen_vecs)
            #gen_vecs = np.append(gen_vecs, dmMass_B, axis = 1)

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
            G_loss_epochs.append(g_loss)
            D_loss_epochs.append(d_loss[0])
            D_acc_epochs.append(d_loss[1])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, scaler, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x, P3z, P1_tot, Px_tot, E_tot, Mass_B, G_loss_epochs, D_loss_epochs, D_acc_epochs)

    def sample_images(self, epoch, scaler, Average_mass_predicted, MPV_mass_predicted, sample_interval, P1x_data, P3z_data, P1_tot_data, Px_tot_data, E_tot_data, Mass_B_data, G_loss_epochs, D_loss_epochs, D_acc_epochs):

        r = 10000;
        noise = np.random.normal(0, 1, (r , self.latent_dim))
        gen_raw = self.generator.predict(noise)
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #scaler = scaler.fit(gen_raw)
        gen_p = scaler.inverse_transform(gen_raw)
        #print(gen_p)
        cnt = 0;
        fig, axs = plt.subplots(3,3)
        fig.set_size_inches(15,14)
        Mass_B = np.zeros(r)
        P1x = np.zeros(r); P3z = np.zeros(r); P1_tot = np.zeros(r); E_tot = np.zeros(r); Px_tot = np. zeros(r)
        for i in range(r):
            p_products = np.array([np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1]) + np.square(gen_p[i][2])),
                       np.sqrt(np.square(gen_p[i][3]) + np.square(gen_p[i][4]) + np.square(gen_p[i][5])),
                       np.sqrt(np.square(gen_p[i][6]) + np.square(gen_p[i][7]) + np.square(gen_p[i][8]))])
            p_total = np.sqrt(np.square(gen_p[i][0]+gen_p[i][3]+gen_p[i][6]) +
                       np.square(gen_p[i][1]+gen_p[i][4]+gen_p[i][7]) +
                       np.square(gen_p[i][2]+gen_p[i][5]+gen_p[i][8]))
            E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
            Mass_B[i] = math.sqrt(np.sum(E_total) ** 2  - p_total ** 2)
            P1x[i] = gen_p[i][0]; P3z[i] = p_total; P1_tot[i] = np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1])
                                                                            + np.square(gen_p[i][2])); E_tot[i] =  np.sum(E_total);
            Px_tot[i] = gen_p[i][0]+gen_p[i][3]+gen_p[i][6]
                #axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                #axs[i,j].axis('off')

        Average_mass_predicted.append(np.mean(Mass_B))

        n, bins, patches = axs[0,0].hist(Mass_B, 200, range =(0,50000), alpha = 0.5, label = 'Generated data')
        #axs[0,0].hist(Mass_B_data[0:10000], 200, range=(0,50000), alpha=0.5, label = 'Input data')
        axs[0,0].set_xlabel('Mass of the B meson [MeV]')
        axs[0,0].set_ylabel('Number of counts')
        axs[0,0].axvline(x=np.mean(Mass_B_data), color ='r', linestyle = 'dashed')
        #axs[0,0].legend(loc='upper right')
        MPV_mass_predicted.append(np.mean(bins[np.where(n == np.amax(n))]))

        n2, bins2, patches2 = axs[0,1].hist(P1x, 200, range = (-100000,100000), alpha = 0.5, label = 'Generated data')
        axs[0,1].hist(P1x_data[0:r], 200, range = (-100000,100000), label = 'Input data', alpha = 0.5)
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_xlabel('Momentum X of K1 [MeV]')
        axs[0,1].set_ylabel('Number of counts')

        axs[1,0].plot(range(0,epoch+sample_interval,sample_interval), Average_mass_predicted, c='r', linewidth=4.0)
        axs[1,0].set_xlim([0,50000])
        axs[1,0].set_ylim([100,500000])
        axs[1,0].set_xlabel('Epoch number')
        axs[1,0].set_ylabel('Mean of the B mass predicted')
        axs[1,0].plot(range(0,epoch+sample_interval,sample_interval), np.zeros(int(epoch/sample_interval) +1)+np.mean(Mass_B_data),'m-.')
        axs[1,0].set_yscale('log')


        axs[1, 1].plot(range(0, epoch + sample_interval, sample_interval), MPV_mass_predicted, c='g', linewidth=4.0)
        axs[1, 1].set_xlim([0, 50000])
        axs[1, 1].set_ylim([100, 500000])
        axs[1, 1].set_xlabel('Epoch number')
        axs[1, 1].set_ylabel('MPV of the B mass predicted')
        axs[1, 1].plot(range(0, epoch + sample_interval, sample_interval), np.zeros(int(epoch/sample_interval) +1)+np.mean(Mass_B_data), 'm-.')
        axs[1, 1].set_yscale('log')
        #fig.savefig("images/%d.png" % epoch)

        axs[2,1].hist(P3z, 200, range = (0,1000000), alpha = 0.5, label = 'Generated data')
        axs[2,1].hist(P3z_data[0:r], 200, range = (0,1000000), label = 'Input data', alpha = 0.5)
        axs[2,1].legend(loc='upper right')
        axs[2,1].set_xlabel('Total momentum [MeV]')
        axs[2,1].set_ylabel('Number of counts')

        axs[2, 0].plot(range(0, epoch + 1, 1), G_loss_epochs, c='g', linewidth=1.0, label = 'G loss')
        axs[2, 0].plot(range(0, epoch + 1, 1), D_loss_epochs, c='r', linewidth=1.0, label = 'D loss')
        axs[2, 0].plot(range(0, epoch + 1, 1), D_acc_epochs, c='c', linewidth=1.0, label = 'D accuracy')
        axs[2, 0].set_xlim([0, 50000])
        axs[2, 0].set_ylim([0, 2])
        axs[2, 0].legend(loc='upper right')
        axs[2, 0].set_xlabel('Epoch number')
        axs[2, 0].set_ylabel('Relative ratio')

        axs[0,2].hist(P1_tot, 200, range = (0,1000000), alpha = 0.5, label = 'Generated data')
        axs[0,2].hist(P1_tot_data[0:r], 200, range = (0,1000000), label = 'Input data', alpha = 0.5)
        axs[0,2].legend(loc='upper right')
        axs[0,2].set_xlabel('Total momentum of K1 [MeV]')
        axs[0,2].set_ylabel('Number of counts')

        axs[1,2].hist(E_tot, 200, range = (0,1000000), alpha = 0.5, label = 'Generated data')
        axs[1,2].hist(E_tot_data[0:r], 200, range = (0,1000000), label = 'Input data', alpha = 0.5)
        axs[1,2].legend(loc='upper right')
        axs[1,2].set_xlabel('Total energy [MeV]')
        axs[1,2].set_ylabel('Number of counts')

        axs[2,2].hist(Px_tot, 200, range = (-200000,200000), alpha = 0.5, label = 'Generated data')
        axs[2,2].hist(Px_tot_data[0:r], 200, range = (-200000,200000), label = 'Input data', alpha = 0.5)
        axs[2,2].legend(loc='upper right')
        axs[2,2].set_xlabel('Total momentum of K3 [MeV]')
        axs[2,2].set_ylabel('Number of counts')

        fig.savefig("images/%d.png" % epoch)

        plt.close()


if __name__ == '__main__':


    gan = GAN()
    gan.train(epochs=50000, batch_size=50, sample_interval=500)
