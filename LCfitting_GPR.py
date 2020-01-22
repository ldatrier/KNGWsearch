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

np.warnings.filterwarnings('ignore')

run_type = 'GPR'

t_0 = 2. #Start in days post-merger
cadence = 1.
filters = ['g','r','i','z','y']

D_L = 44. #DISTANCE TO SOURCE

GPs = {}
for x in filters:
    GPs[x]=pickle.load(open('DECam_GPR/gpr_model_dc_{}.sav'.format(x),'rb'))

Norms = {}
for x in filters:
    #Extracting normalisation factors, with [0] as max and [1] as min
    filename = "DECam_GPR/norms_dc_{}.txt".format(x)
    with open(filename) as f:
        fdata = f.read()
    Norms[x] = re.findall(r'[+-]?\d+.\d+',fdata)


def app_m(D,M):
    return M-5+5*np.log10(D*10**6)

time_index = np.arange(t_0,10,cadence)
Truth = {'t0':t_0, 'DL':D_L,'mej':0.03,'mejb':0.02,'vej':0.1,'vejb':0.22,'Xlan':1.6,'Xlanb':8}

observations = {}

for x in filters:
    xtest_b = np.array([[time_index[y],np.log10(Truth['mejb']),np.log10(Truth['vejb']),Truth['Xlanb']] for y,t in enumerate(time_index)])
    xtest_r = np.array([[time_index[y],np.log10(Truth['mej']),np.log10(Truth['vej']),Truth['Xlan']] for y,t in enumerate(time_index)])
    y_pred_blue, sigma_blue = GPs[x].predict(xtest_b,return_std=True)
    y_pred_red, sigma_red = GPs[x].predict(xtest_r,return_std=True)
    KNLC =  -2.5*np.log10(10**(-(y_pred_blue * float(Norms[x][0]) + float(Norms[x][1]))*0.4) + 10**(-(y_pred_red * float(Norms[x][0]) + float(Norms[x][1]))*0.4))
    observations[x] = app_m(D_L,KNLC)

time_index = time_index - t_0

def like_single(args):
    t0, DL, mej, vej, Xlan, mej_b, vej_b, Xlan_b = args
    filter_sum_prob = {}
    KN_modeltest = {}
    for x in filters:
        index = time_index + t0
        xtestb = np.array([[index[y],np.log10(mej_b),np.log10(vej_b),Xlan_b] for y,t in enumerate(index)])
        xtestr = np.array([[index[y],np.log10(mej),np.log10(vej),Xlan] for y,t in enumerate(index)])
        ypredblue, sigmablue = GPs[x].predict(xtestb,return_std=True)
        ypredred, sigmared = GPs[x].predict(xtestr,return_std=True)
        kn =  -2.5*np.log10(10**(-(ypredblue * float(Norms[x][0]) + float(Norms[x][1]))*0.4) + 10**(-(ypredred * float(Norms[x][0]) + float(Norms[x][1]))*0.4))
        #sigmas = .2
        sigmas =  np.sqrt(sigmablue**2 + sigmared**2) * float(Norms[x][0])
        KN_modeltest[x] = app_m(DL,kn)
        filter_sum_prob[x] = -np.sum((KN_modeltest[x]-observations[x])**2/sigmas**2)
        #filter_sum_prob[x] = np.sum(norm.logpdf(KN_modeltest[x],loc=observations[x],scale=sigmas))
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
priors['t0'] = bilby.prior.Uniform(minimum=0.,maximum=5,name='t0')
priors['mej'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej')
priors['vej'] = bilby.prior.Uniform(minimum=0.05,maximum=0.35,name='vej')
priors['Xlan'] = bilby.prior.Uniform(minimum=1.,maximum=2.,name='Xlan')
priors['mej_b'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej_b')
priors['vej_b'] = bilby.prior.Uniform(minimum=0.05,maximum=0.35,name='vej_b')
priors['Xlan_b'] = bilby.prior.Uniform(minimum=2.,maximum=9.,name='Xlan_b')
priors['vfrac'] = bilby.prior.Constraint(minimum=0.,maximum=0.25,name='vfrac')

print('About to run sampler')
print(datetime.now())

now = datetime.now()
l = '{}'.format(now.strftime("%Y-%m-%d-%H%M"))
rpath = 'Results/{}/{}'.format(run_type,l)


os.mkdir(rpath)
shutil.copy('calctdl_GPR.py','Results/{}/{}/src_code.py'.format(run_type,l),follow_symlinks=True)


sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='dynesty', outdir='Results/{}/{}'.format(run_type,l),label=l,verbose = True, dlogz = .1,npoints = 2000, n_check_point = 3000 )

print('Run complete')
print(datetime.now())

np.savetxt('Results/{}/{}/posterior'.format(run_type,l),np.array(sampler.posterior))

sampler.plot_corner(truth={'t0':t_0,'D_L':44,'mej':Truth['mej'],'vej':Truth['vej'],'Xlan':Truth['Xlan'],'mej_b':Truth['mejb'],'vej_b':Truth['vejb'],'Xlan_b':Truth['Xlanb']})


