import numpy as np
from matplotlib import pyplot as plt
import os,random,glob
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
import re
from datetime import datetime
import pickle
import argparse
#import pymc3

i = np.load("KN_r.npy")
y = np.array([h[-1] for h in i])
x = np.array([h[0:4] for h in i])
filter = 'r'
#This is a mess - needs rewriting!
x_array = np.asarray(x)
y_array = np.asarray(y)
cross_indices = np.where((x_array[:,1:4]==(0.1, 0.03,5.)).all(axis=1))
x_crossval = x_array[np.where((x_array[:,1:4]==(0.1, 0.03,5.)).all(axis=1))]
x_training = np.delete(x_array,cross_indices,axis=0)
y_crossval = y_array[np.where((x_array[:,1:4]==(0.1, 0.03,5.)).all(axis=1))]
y_training = np.delete(y_array,cross_indices,axis=0)
x_training = x_training[::50]
x_training[:,1:3] = np.log10(x_training[:,1:3])
y_training = y_training[::50]
y_norm_min = np.min(y_training)
y_training = (y_training - y_norm_min)
y_norm_max = np.max(y_training)
y_training = y_training / y_norm_max
print('About to run GPR')
print(datetime.now())

# Instanciate a Gaussian Process model
kernel =  RationalQuadratic(4.,1.) + WhiteKernel(1.)#,(1e-4,1.0))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,normalize_y=False)
gp.fit(x_training,y_training)
print('Run complete')
print(datetime.now())
pickle.dump(gp, open('gpr_model_{}.sav'.format(filter), 'wb'))

mej_test = np.log10(0.1)
vej_test = np.log10(0.03)
Xlan_test = 5.
testindex = np.linspace(0,30,100)
x_test = np.array([[testindex[i], mej_test, vej_test, Xlan_test] for i,t in enumerate(testindex)])
y_pred, sigma = gp.predict(x_test,return_std = True)
y_pred = (y_pred * y_norm_max) + y_norm_min


plt.plot(testindex,y_pred, 'b', lw=2)
plt.plot(x_crossval[:,0],y_crossval,'r',lw=3)
plt.fill_between(testindex, (y_pred - 2*sigma), (y_pred + 2*sigma), color="b", alpha=0.2)
plt.fill_between(testindex , (y_pred - 3*sigma), (y_pred + 3*sigma), color="b", alpha=0.1)
plt.fill_between(testindex , (y_pred - 5*sigma), (y_pred + 5*sigma), color="b", alpha=0.05)
plt.gca().invert_yaxis()
plt.show()

y_samples = gp.sample_y(x_test,12)
plt.plot(testindex,y_samples, 'b',alpha=.6)
plt.show()


