import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from gaussfft import gaussfft

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave

ex = 2.3

if ex==0:
	print("That was a stupid idea")

if ex==1.3:
	# Exercise 1.3

	fftwave(5, 9, sz=128)
	fftwave(9, 5, sz=128)
	fftwave(17, 9, sz=128)
	fftwave(17, 121, sz=128)
	fftwave(5, 1, sz=128)
	fftwave(125, 1, sz=128)
	print()

if ex==1.4:
	# Exercise 1.4: Linearity

	# Create images
	F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
	G = F.T
	H = F + 2 * G
	showgrey(F, display=False)
	showgrey(G, display=False)
	showgrey(H, display=False)

	# Compute discrete fourier transform and show fourier spectra
	Fhat = fft2(F)
	Ghat = fft2(G)
	Hhat = fft2(H)

	showgrey(np.log(1 + np.abs(Fhat)), display=False)
	showgrey(np.log(1 + np.abs(Ghat)), display=False)
	showgrey(np.log(1 + np.abs(Hhat)), display=False)

	# Centering: (the same can be achieved with the function showfs)
	showgrey(np.log(1 + np.abs(fftshift(Hhat))), display=False)

if ex==1.5:
	# Exercise 1.5: Multiplication
	F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
	G = F.T
	showgrey(F * G, display=False)
	showfs(fft2(F * G), display=False)

if ex==1.6:
	# Exercise 1.6: Scaling

	F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
		np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)

	showgrey(F, display=False)
	showfs(fft2(F), display=False)

if ex==1.7:
	# Exercise 1.7: Rotation
	f = plt.figure()

	F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
		np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
	alpha = 90
	G = rot(F, alpha)

	a1 = f.add_subplot(1, 3, 1)
	showgrey(G, display=False)
	a1.title.set_text("alpha = %d"%(alpha))

	Ghat = fft2(G)
	a2 = f.add_subplot(1, 3, 2)
	showfs(Ghat, display=False)
	a2.title.set_text("fft")

	Hhat = rot(fftshift(Ghat), -alpha)
	a3 = f.add_subplot(1, 3, 3)
	showgrey(np.log(1 + abs(Hhat)), display=False)
	a3.title.set_text("Back")
	plt.show()
	print()

if ex==1.8:
	# Exercise 1.8: Information in Fourier Phase and Magnitude
	images = ["Images-npy/phonecalc128.npy", "Images-npy/few128.npy", "Images-npy/nallo128.npy"]
	img = np.load(images[0])

	f, ax = plt.subplots(1, 2)
	ax[0].imshow(img)
	ax[1].imshow(pow2image(img))
	plt.show()
	print()

if ex==2.3:
	test = discgaussfft(deltafcn(128, 128), 0.1)
	# img = gaussfft(deltafcn(128, 128), t=0.1)
	var = variance(img)
	print()
print()
