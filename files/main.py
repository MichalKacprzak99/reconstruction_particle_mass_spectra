from GAN.gan import GAN
from BGAN.bgan import BGAN
from WGAN.wgan import WGAN
from DUALGAN.dualgan import DUALGAN
from COGAN.cogan import COGAN
if __name__ == '__main__':
    dualgan = DUALGAN()
    dualgan.train(1000,128,5)
