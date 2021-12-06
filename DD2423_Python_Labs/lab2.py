import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft

# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

ex = 2


def deltax(dir=1):
        dxmask = dir*np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        return dxmask


def deltay(dir=1):
        dymask = dir*np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return dymask


def Lv(inpic, shape='same', smooth=True, scale=0.1, type=3):
        if type ==0:
                dxmask = np.asarray([-1, 0, 1])
                dymask = np.asarray([-1, 0, 1]).transpose()
        if type == 1:
                dxmask = np.asarray([-0.5, 0, 0.5])
                dymask = np.asarray([-0.5, 0, 0.5]).transpose()
        if type == 2:
                dxmask = np.asarray([[-1, 0], [0, 1]])
                dymask = np.asarray([[0, -1], [1, 0]])
        if type == 3:
                dxmask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
                dymask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        if smooth == True:
                inpic = discgaussfft(inpic, scale)

        Lx = convolve2d(inpic, dxmask, shape)
        Ly = convolve2d(inpic, dymask, shape)
        return np.sqrt(Lx ** 2 + Ly ** 2)
#
#
def Lvvtilde(inpic, shape='same'):
        dxmask = np.zeros((5, 5))
        dxmask[2,1:4] = [0.5, 0, -0.5]
        dymask = dxmask.transpose()

        dxxmask = np.zeros((5, 5))
        dxxmask[2, 1:4] = [1, -2, 1]
        dyymask = dxxmask.transpose()

        dxymask = convolve2d(dxmask, dymask, shape)

        Lx = convolve2d(inpic, dxmask, shape)
        Ly = convolve2d(inpic, dymask, shape)
        Lxx = convolve2d(inpic, dxxmask, shape)
        Lxy = convolve2d(inpic, dxymask, shape)
        Lyy = convolve2d(inpic, dyymask, shape)

        result = ((Lx**2) * Lxx + 2*Lx*Ly*Lxy + (Ly**2)*Lyy)
        return result


def Lvvvtilde(inpic, shape='same'):
        dxmask = np.zeros((5, 5))
        dxmask[2,1:4] = [0.5, 0, -0.5]
        dymask = dxmask.transpose()

        dxxmask = np.zeros((5, 5))
        dxxmask[2, 1:4] = [1, -2, 1]
        dyymask = dxxmask.transpose()

        dxyymask = convolve2d(dxmask, dyymask, shape)
        dxxymask = convolve2d(dxxmask, dymask, shape)

        dxxxmask = convolve2d(dxxmask, dxmask, shape)
        dyyymask = convolve2d(dyymask, dymask, shape)

        Lx = convolve2d(inpic, dxmask, shape)
        Ly = convolve2d(inpic, dymask, shape)
        Lxxy = convolve2d(inpic, dxxymask, shape)
        Lxyy = convolve2d(inpic, dxyymask, shape)
        Lxxx = convolve2d(inpic, dxxxmask, shape)
        Lyyy = convolve2d(inpic, dyyymask, shape)

        result = ((Lx ** 3) * Lxxx + 3 * (Lx**2) * Ly * Lxxy + 3 * Lx * (Ly**2) * Lxyy + (Ly ** 3) * Lyyy)
        return result


def extractedge(inpic, scale, shape='same', threshold=None):
        Lvv = Lvvtilde(discgaussfft(inpic, scale), shape)
        mask1 = (Lvvvtilde(discgaussfft(inpic, scale), shape) < 0)

        edgecurves = zerocrosscurves(Lvv, mask1)
        if threshold is not None:
                mask2 = (Lv(inpic, shape='same', smooth=True, scale=0.1) > threshold)
                edgecurves = thresholdcurves(edgecurves, mask2)

        return edgecurves


def houghline(curves, magnitude, nrho, ntheta, threshold, nlines=20, verbose=False):
        # Allocate accumulator space
        acc = np.zeros((nrho, ntheta))

        # Define a coordinate system in the accumulator space
        max_rho = np.sqrt(magnitude.shape[0] ** 2 + magnitude.shape[1] ** 2)
        rho_space = np.linspace(-max_rho, max_rho, nrho)
        theta_space = np.linspace(0., np.pi, ntheta)


        # Loop over all the edge points
        for x, y in zip(curves[0], curves[1]):
                # Check if valid point with respect to threshold
                # Optionally, keep value from magnitude image

                # Loop over a set of theta values
                for idx_theta, theta in enumerate(theta_space):
                        # Compute rho for each theta value
                        rho = x * np.cos(theta) + y * np.sin(theta)
                        # Compute index values in the accumulator space
                        idx_rho = np.argmin(np.abs(rho_space-rho))
                        # Update the accumulator
                        acc[idx_rho, idx_theta] += np.abs(magnitude[x, y])/np.max(magnitude)

        # TODO: smooth before compute maxima
        # Extract local maxima from the accumulator
        pos, value, _ = locmax8(acc)
        indexvector = np.argsort(value)[-nlines:]
        pos = pos[indexvector]

        # extract index values in the accumulator
        linepar = np.zeros((2, nlines))
        for idx in range(nlines):
                rhoidxacc = pos[idx, 1]
                thetaidxacc = pos[idx, 0]

                linepar[0, idx] = rho_space[rhoidxacc]
                linepar[1, idx] =  1e-6 if theta_space[thetaidxacc] == 0 else theta_space[thetaidxacc]



        # Delimit the number of responses if necessary

        # Compute a line for each one of the strongest responses in the accumulator


        # Overlay these curves on the gradient magnitude image
        # Return the output data [linepar, acc]
        return linepar, acc


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines=20, verbose=False):
        edgecurves = extractedge(pic, scale, shape='same', threshold=gradmagnthreshold)
        magnitude = Lv(pic, shape='same', smooth=True, scale=0.1, type=3)

        linepar, acc = houghline(edgecurves, magnitude, nrho, ntheta, gradmagnthreshold, nlines, verbose)
        f = plt.figure()
        showgrey(pic, False)
        max_rho = np.sqrt(pic.shape[0] **2 + pic.shape[1] ** 2)
        for i in range(linepar.shape[1]):
                rho = linepar[0,i]
                theta = linepar[1,i]

                x0 = max_rho/2
                y0 = (-np.cos(theta)/np.sin(theta)) * x0 + (rho/np.sin(theta))
                dx = max_rho ** 2
                # dy = (rho - np.cos(theta) * dx) / np.sin(theta)
                dy = (rho - (x0 + dx)*np.cos(theta))/np.sin(theta) - y0
                plt.plot([y0 - dy, y0, y0 + dy], [x0 - dx, x0, x0 + dx], 'r-')

        plt.show()


if ex ==2:
        images = ["Images-npy/few256.npy", "Images-npy/godthem256.npy"]
        tools = np.load(images[1])

        dxtools = convolve2d(tools, deltax(), 'valid')
        dytools = convolve2d(tools, deltay(), 'valid')

        dxtools_inv = convolve2d(tools, deltax(-1), 'valid')
        dytools_inv = convolve2d(tools, deltay(-1), 'valid')

        f = plt.figure()
        f.add_subplot(2,3,1)
        showgrey(tools, False)
        f.add_subplot(2,3,2)
        showgrey(dxtools, False)
        f.add_subplot(2,3,3)
        showgrey(dytools, False)
        f.add_subplot(2,3,4)
        showgrey(tools, False)
        f.add_subplot(2,3,5)
        showgrey(dxtools_inv, False)
        f.add_subplot(2,3,6)
        showgrey(dytools_inv, False)

        # check sizes
        #print("Size tools: " + str(tools.shape) + " -  size dxTools: " + str(dxtools.shape))

        plt.show()

        # Compute gradient magnitude
        gradmagntools = np.sqrt(dxtools**2 + dytools**2)
        # showgrey(gradmagntools, False)

        plt.hist(np.reshape(gradmagntools, [-1]), 1000)
        plt.show()

        f = plt.figure()
        threshold = 200
        f.add_subplot(1, 2, 1)
        showgrey((gradmagntools > threshold).astype(int), False)

        # Smoothing
        smoth_tools = discgaussfft(tools, 1.)
        f.add_subplot(1, 2, 2)
        showgrey(((Lv(smoth_tools)> threshold).astype(int)), False)

        plt.show()

if ex == 4:

        # Test masks: OK
        # [x, y] = np.meshgrid(range(-5, 6), range(-5, 6))
        #
        # dxmask = np.zeros((5, 5))
        # dxmask[2, 1:4] = [0.5, 0, -0.5]
        # dymask = dxmask.transpose()
        #
        # dxxmask = np.zeros((5, 5))
        # dxxmask[2, 1:4] = [1, -2, 1]
        # dyymask = dxxmask.transpose()
        #
        # dxxxmask = convolve2d(dxmask, dxxmask, 'same')
        # dxxymask = convolve2d(dxxmask, dymask, 'same')
        # dyyymask = convolve2d(dyymask, dymask, 'same')
        #
        # print(convolve2d(x ** 3, dxxxmask, 'valid'))
        # print(convolve2d(x ** 3, dxxmask, 'valid'))
        # print(convolve2d(x ** 2 * y, dxxymask, 'valid'))

        ############# QUESTION 1 ###############

        house = np.load("Images-npy/godthem256.npy")
        tools = np.load("Images-npy/few256.npy")

        scales = [0.0001, 1.0, 4.0, 16.0, 64.0]
        f = plt.figure()
        N = 5
        n = 1
        for scale in scales:
                f.add_subplot(2, 5, n)
                showgrey(contour(Lvvtilde(discgaussfft(tools, scale), 'same')), False)

                f.add_subplot(2, 5, N + n)
                showgrey((Lvvvtilde(discgaussfft(tools, scale), 'same') < 0).astype(int), False)
                n+=1


        plt.show()

if ex == 5:

        house = np.load("Images-npy/godthem256.npy")
        tools = np.load("Images-npy/few256.npy")

        scales = [0.0001, 1.0, 4.0, 16.0, 64.0]
        f = plt.figure()
        N = 5
        n = 1
        for scale in scales:
                f.add_subplot(2, 5, n)
                edgecurves1 = extractedge(house, scale, threshold=200)
                overlaycurves(house, edgecurves1)

                f.add_subplot(2, 5, N + n)
                edgecurves2 = extractedge(tools, scale, threshold=200)
                overlaycurves(tools, edgecurves2)
                n += 1


        plt.show()


if ex == 6:
        # testimage1 = np.load("Images-npy/triangle128.npy")
        testimage1 = np.load("Images-npy/godthem256.npy")
        smalltest1 = binsubsample(testimage1)
        houghedgeline(smalltest1, scale=1., gradmagnthreshold=10, nrho=200, ntheta=180, nlines=20, verbose=False)
        print()






print()



