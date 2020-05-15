import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
import os,glob,re
import pandas as pd
import random
import corner
from datetime import datetime
import bilby
from scipy.special import logsumexp
import pickle
import shutil
from george import kernels
import george

np.warnings.filterwarnings('ignore')


t_0 = 0. #Start in days post-merger
cadence = 1.
D_L = 40.

filters = ['g','r','i']
GPs = {}
y_training = {}
#kernel = 3.1*kernels.RationalQuadraticKernel(log_alpha=.5,metric=[3.0, .001, .01, 1.0], ndim=4)#([2.0, .25, .2, 1.0], ndim=4)
kernel = 1. * kernels.ExpSquaredKernel([4., .02, .05, .5], ndim=4)
data = {}
yy = {}
xx = {}
x_training = {}
y_training = {}

for x in filters:
    data[x] = np.load("Smooth_models_DECam/KN_{}_dc.npy".format(x))
    yy[x] = np.array(data[x][:,-1])
    xx[x] = np.array([data[x][:,0],data[x][:,1],data[x][:,2],data[x][:,3]]).T
    #xx[x][:,1:3] = np.log10(xx[x][:,1:3])
    yy[x] = np.delete(yy[x],np.where(xx[x][:,-1]==9.),axis=0)
    xx[x] = np.delete(xx[x],np.where(xx[x][:,-1]==9.),axis=0)
    x_training[x] = xx[x][::30]#[:5000]
    y_training[x] = yy[x][::30]#[:5000]
    print('About to run GPR on {} band'.format(x))
    print(datetime.now())
    GPs[x] = george.GP(kernel, solver=george.HODLRSolver, tol=1e-5)
    GPs[x].compute(x_training[x], .5*np.ones(len(x_training[x])))
    print('Run complete')
    print(datetime.now())


run_type = 'GPR_george'
def app_m(D,M):
    return M-5+5*np.log10(D*10**6)

timeindex = np.arange(t_0,10,cadence)
Truth = {'t0':t_0, 'DL':D_L,'mej':0.03,'mejb':0.02,'vej':0.1,'vejb':0.22,'Xlan':1.6,'Xlanb':3.7}
truth = [t_0,D_L,0.03,0.1,1.6,0.02,0.22,3.7]

observations = {}
time_index = {}
for x in filters:
    xtest_b = np.array([[timeindex[y],Truth['mejb'],Truth['vejb'],Truth['Xlanb']] for y,t in enumerate(timeindex)])
    xtest_r = np.array([[timeindex[y],Truth['mej'],Truth['vej'],Truth['Xlan']] for y,t in enumerate(timeindex)])
    y_pred_blue, sigma_blue = GPs[x].predict(y_training[x],xtest_b,return_var=True)
    y_pred_red, sigma_red = GPs[x].predict(y_training[x],xtest_r,return_var=True)
    KNLC =  -2.5*np.log10(10**(-y_pred_blue *0.4) + 10**(-y_pred_red *0.4))
    observations[x] = app_m(D_L,KNLC)
    time_index[x] = timeindex - t_0



def like_single(args):
	t0, DL, mej, vej, Xlan, mej_b, vej_b, Xlan_b = args
	filter_sum_prob = {}
	KN_modeltest = {}
	for x in filters:
		index = time_index[x] + t0
		xtestr = np.array([[index[y],mej,vej,Xlan] for y,t in enumerate(index)])
		xtestb = np.array([[index[y],mej_b,vej_b,Xlan_b] for y,t in enumerate(index)])
		ypredblue, sigmablue = GPs[x].predict(y_training[x],xtestb,return_var=True)
		ypredred, sigmared = GPs[x].predict(y_training[x],xtestr,return_var=True)
		kn =  -2.5*np.log10(10**(-ypredblue*0.4) + 10**(-ypredred*0.4))
		sigmas =  np.sqrt(sigmablue**2 + sigmared**2)
		KN_modeltest[x] = app_m(DL,kn)
		filter_sum_prob[x] = logsumexp(norm.logpdf(KN_modeltest[x],loc=observations[x],scale=sigmas))
	return np.sum([filter_sum_prob[x] for x in filters])


class KNmodelprob(bilby.Likelihood):
    def __init__(self, nthreads=None):
        self.parameters = {'t0':None,'D_L':None,'mej': None, 'vej': None, 'Xlan': None,'mej_b': None, 'vej_b': None, 'Xlan_b': None, 'vfrac':None}
    def log_likelihood(self):
        D_L = self.parameters['D_L']
        t0 = self.parameters['t0']
        mej = self.parameters['mej']
        vej = self.parameters['vej']
        Xlan = self.parameters['Xlan']
        mej_b = self.parameters['mej_b']
        vej_b = self.parameters['vej_b']
        Xlan_b = self.parameters['Xlan_b']
        vfrac = self.parameters['vfrac']
        args = [t0, D_L, mej, vej, Xlan, mej_b, vej_b, Xlan_b]
        prob = like_single(args)
        return prob


def convert_x_y_to_z(parameters):
    parameters['vfrac'] = parameters['vej_b'] - parameters['vej']
    return parameters

priors = bilby.core.prior.PriorDict(conversion_function=convert_x_y_to_z)

priors['D_L'] = bilby.prior.Uniform(minimum=35,maximum=49,name='D_L')
priors['t0'] = bilby.prior.Uniform(minimum=0.,maximum=8.,name='t0')
priors['mej'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej')
priors['vej'] = bilby.prior.Uniform(minimum=0.05,maximum=0.35,name='vej')
priors['Xlan'] = bilby.prior.Uniform(minimum=1.,maximum=2.,name='Xlan')
priors['mej_b'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej_b')
priors['vej_b'] = bilby.prior.Uniform(minimum=0.05,maximum=0.35,name='vej_b')
priors['Xlan_b'] = bilby.prior.Uniform(minimum=2.,maximum=5.,name='Xlan_b')
priors['vfrac'] = bilby.prior.Constraint(minimum=0.,maximum=0.25,name='vfrac')


print('About to run sampler')
print(datetime.now())

now = datetime.now()
l = '{}'.format(now.strftime("%Y-%m-%d-%H%M"))
rpath = 'Results/{}/{}'.format(run_type,l)


os.mkdir(rpath)
#shutil.copy('calctdl_GPRgen_george.py','Results/{}/{}/src_code.py'.format(run_type,l),follow_symlinks=True)


sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='dynesty', outdir='Results/{}/{}'.format(run_type,l),label=l,verbose = True, dlogz = .1,npoints = 2000, n_check_point = 3000 )

print('Run complete')
print(datetime.now())

np.savetxt('Results/{}/{}/posterior'.format(run_type,l),np.array(sampler.posterior))

sampler.plot_corner(truth={'t0':t_0,'D_L':40,'mej':Truth['mej'],'vej':Truth['vej'],'Xlan':Truth['Xlan'],'mej_b':Truth['mejb'],'vej_b':Truth['vejb'],'Xlan_b':Truth['Xlanb']})
