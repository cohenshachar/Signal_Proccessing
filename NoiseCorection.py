import numpy as np
from PIL import Image as image
import pandas as pd

N = 256 # samples

# load and create pixel arrays for further mod
img_grey = (image.open("greyScaleImage256by256.gif")).convert("L")
pixels = [np.array(img_grey).astype('float')]
for _ in range(3):
    pixels.append(pixels[0].copy())

# create noises
n = np.arange(N)
j = n.reshape((N, 1))

A, phi, R= [], [], []
f =[1/8,1/32]
for k in range(2):
    A.append(np.random.normal(0.1, 0.05, N))
    phi.append(np.random.uniform(0, 2*np.pi, N))
    R.append(np.array([
        [255 * A[k][i] * np.cos(2*np.pi*f[k]*j+phi[k][i]) for j in n] for i in n
    ]))

# apply noises

pixels[1] += R[0]
pixels[2] += R[1]
pixels[3] += (R[0] + R[1])/2

# construct images and calculate MSE
MSE=[0]
for i in range(1,4):
    pixels[i] = pixels[i].clip(0,255).astype('uint8')
    image.fromarray(pixels[i], 'L').save("I"+str(i)+".png")
    MSE.append(pixels[0] - pixels[i])
    MSE[i] *= MSE[i]
    MSE[i] = np.sum(MSE[i]) / (255 ** 2)
    pixels[i] = pixels[i].astype('float64') / 255
print(MSE)

pixels[0] = pixels[0].astype('float64') / 255

# construct dft
dft = np.exp(-2j * np.pi * j * n / N) / np.sqrt(N)
f_location = [int(N*f[0]), int(N*f[1])]

# apply dft
discrete_signals_per_row = []
for i in range(4):
    discrete_signals_per_row.append(pixels[i] @ dft)
    pd.DataFrame(discrete_signals_per_row[i]).to_csv('representative'+str(i)+'.csv', index=False)

# calculate the diff from the theory
for i in range(0,4):
    (pd.DataFrame(np.round( np.sum(np.abs(discrete_signals_per_row[0] - discrete_signals_per_row[i]),axis=0),decimals = 0))
     .to_csv('cmp_representative'+str(i)+'.csv', index=False))

pixels[0] *= 255

# 0-out noise frequencies
discrete_signals_per_row[1][:,f_location[0]] = 0 +0j
discrete_signals_per_row[2][:,f_location[1]] = 0 +0j
discrete_signals_per_row[3][:,f_location[0]] = 0 +0j
discrete_signals_per_row[3][:,f_location[1]] = 0 +0j

discrete_signals_per_row[1][:,N-f_location[0]] = 0 +0j
discrete_signals_per_row[2][:,N-f_location[1]] = 0 +0j
discrete_signals_per_row[3][:,N-f_location[0]] = 0 +0j
discrete_signals_per_row[3][:,N-f_location[1]] = 0 +0j

# revert back to greyscale-world, construct images and calculate MSE
MSE=[0]
for i in range(1,4):
    pixels[i] = ((discrete_signals_per_row[i] @ np.conjugate(dft)) * 255).astype('uint8')
    image.fromarray(pixels[i], 'L').save("I"+str(i)+"_reconstruct.png")
    MSE.append(pixels[0] - pixels[i])
    MSE[i] *= MSE[i]
    MSE[i] = np.sum(MSE[i]) / (255 ** 2)
print(MSE)


