import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
from datetime import datetime
import george


filter = 'i'
i = np.load("Smooth_models/KN_{}.npy".format(filter))
y = np.array([h[-1] for h in i])
x = np.array([h[0:4] for h in i])


#x_array = np.asarray(x)
#y_array = np.asarray(y)
cross_indices = np.where((x_array[:,1:4]==(0.1, 0.03,5.)).all(axis=1))
x_crossval = x_array[np.where((x_array[:,1:4]==(0.1, 0.03,5.)).all(axis=1))]
#x_training = np.delete(x_array,cross_indices,axis=0)
y_crossval = y_array[np.where((x_array[:,1:4]==(0.1, 0.03,5.)).all(axis=1))]
#y_training = np.delete(y_array,cross_indices,axis=0)

x_training = np.asarray(x)
y_training = np.asarray(y)

x_training = x_training[::20]#[:5000]
x_training[:,1:3] = np.log10(x_training[:,1:3])
y_training = y_training[::20]#[:5000]

print('About to run GPR')
print(datetime.now())

from george import kernels
kernel =  kernels.ExpSquaredKernel([2.0, .25, .2, 1.0], ndim=4)
gp_hodlr = george.GP(kernel, solver=george.HODLRSolver, tol=1e-5)
gp_hodlr.compute(x_training, .2*np.ones(len(x_training)))
print('Run complete')
print(datetime.now())


mej_test = np.log10(0.1)
vej_test = np.log10(0.03)
Xlan_test = 5.
testindex = np.linspace(0,30,100)
x_test = np.array([[testindex[i], mej_test, vej_test, Xlan_test] for i,t in enumerate(testindex)])
y_pred, sigma = gp_hodlr.predict(y_training,x_test,return_var = True)


plt.plot(testindex,y_pred, 'b', lw=2)
plt.plot(x_crossval[:,0],y_crossval,'r',lw=3)
plt.fill_between(testindex, (y_pred - 2*sigma), (y_pred + 2*sigma), color="b", alpha=0.2)
plt.fill_between(testindex , (y_pred - 3*sigma), (y_pred + 3*sigma), color="b", alpha=0.1)
plt.fill_between(testindex , (y_pred - 5*sigma), (y_pred + 5*sigma), color="b", alpha=0.05)
plt.gca().invert_yaxis()
plt.show()


