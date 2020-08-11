from GAN.gan import GAN
from BGAN.bgan import BGAN
from WGAN.wgan import WGAN
from DUALGAN.dualgan import DUALGAN
from COGAN.cogan import COGAN
if __name__ == '__main__':
    cogan = COGAN()
    # wgan.train(epochs=30000, batch_size=512, sample_interval=100)
