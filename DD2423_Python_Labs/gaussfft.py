import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from Functions import *

def gaussfft(pic, t, display=False):
    # pic[10,10] = 1



    pfft = np.fft.fft2(pic)
    # pfft = fftshift(np.fft.fft2(pic))


    # pfft_shift = fftshift(pfft)
    # a3 = f.add_subplot(2, 3, 3)
    # # showgrey(pfft_shift, display=False)
    # showfs(pfft, display=False)
    # a3.title.set_text("FFT Shift")



    [h, w] = np.shape(pic)
    # [x, y] = np.meshgrid(np.linspace(0, 1 - 1 / w, w), np.linspace(0, 1 - 1 / h, h))
    [x, y] = np.meshgrid(np.linspace(-w/2,  w/2-1 , w), np.linspace(- h/2,  h/2 - 1, h))
    # [x, y] = np.meshgrid(np.linspace(-(w-1) / 2, (w-1) / 2, w), np.linspace(-(h-1) / 2, (h-1) / 2, h))
    # [x1, y1] = np.meshgrid(np.linspace(0, (1 - 1 / w), w), np.linspace(0, (1 - 1 / h), h))

    gauss = np.exp((-(x**2 + y**2) / float(2 * t)))/(2*t*np.pi)
    # gauss1 = np.exp((-(x1 ** 2 + y1 ** 2) / float(2 * t)))



    # Fourier and shift
    G_hat = fftshift(fft2(gauss))
    # G_hat_1 = fftshift(fft2(gauss1))

    # Shift and fourier
    G_hat_2 = np.fft.fftshift(gauss)
    # G_hat_3 = np.fft.fftshift(gauss1)

    # Fourier
    G_hat_4 = fft2(fftshift(gauss))       # TODO: this is the right one
    # G_hat_5 = fft2(gauss1)

    img = pfft * G_hat_4
    # img = pfft * fftshift(G_hat_4)
    # img = pfft * G_hat_5

    # img = pfft_shift * G_hat_2
    # img = pfft * G_hat_2


    final = ifft2(img)


    if display:
        f = plt.figure()
        a1 = f.add_subplot(2, 3, 1)
        showgrey(pic, display=False)
        a1.title.set_text("Image")

        a2 = f.add_subplot(2, 3, 2)
        showfs(pfft, display=False)
        a2.title.set_text("FFT ")

        a3 = f.add_subplot(2, 3, 3)
        showgrey(gauss, display=False)
        a3.title.set_text("Gauss filter")

        a4 = f.add_subplot(2, 3, 4)
        # showfs(G_hat_4, display=False)
        showfs(G_hat_4, display=False)
        a4.title.set_text("Fourier Gauss")

        a5 = f.add_subplot(2, 3, 5)
        showfs(img, display=False)
        # showgrey(img)
        a5.title.set_text("Fourier Multiplication")

        a6 = f.add_subplot(2, 3, 6)
        showgrey(np.real(final), display=False)
        a6.title.set_text("Final image")

    ############# TEST ###############
    # test = plt.figure()
    # t1 = test.add_subplot(2, 2, 1)
    # showgrey(gauss, display=False)
    # t1.title.set_text("gauss  centered")
    #
    # t2 = test.add_subplot(2, 2, 2)
    # showgrey(gauss1, display=False)
    # t2.title.set_text("gauss")
    #
    # t3 = test.add_subplot(2, 2, 3)
    # showfs(G_hat_4, display=False)
    # # showgrey(G_hat, display=False)
    # t3.title.set_text("Fourier Gauss Centered")
    #
    # t4 = test.add_subplot(2, 2, 4)
    # showfs(G_hat_5, display=False)
    # # showgrey(G_hat_1, display=False)
    # t4.title.set_text("Fourier Gauss")

    ##################################################

    #plt.show()

    return np.real(final)

# gaussfft(deltafcn(128,128), 10.0)
