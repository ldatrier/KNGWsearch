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

run_type = 'gw170817'

t_0 = 11.3888399999 / 24. #Start in days post-merger
true_t0 = 57982.528524 #Actual start day
filters = ['g','r','i','z','y']

D_L = 44. #DISTANCE TO SOURCE

GPs = {}
for x in filters:
    GPs[x]=pickle.load(open('DECam_GPR/gpr_model_dc_{}.sav'.format(x),'rb'))

Norms = {}
for x in filters:
    #Extracting normalisation factors, with [0] as max and [1] as min
    filename = "/DECam_GPR/norms_dc_{}.txt".format(x)
    with open(filename) as f:
        fdata = f.read()
    Norms[x] = re.findall(r'[+-]?\d+.\d+',fdata)


def app_m(D,M):
    return M-5+5*np.log10(D*10**6)

"""
Opening gw170817 DECam observations
"""
KN = pd.read_table("gw170817.txt",delimiter=" ")

KN_obs = KN.pivot(index=KN['DATE'],columns='FILTER')['MAG']
KN_obs.rename(str.lower,axis='columns',inplace=True)
KN_obs.index = KN_obs.index  - true_t0
KN_error = KN.pivot(index=KN['DATE'],columns='FILTER')['ERR']
KN_error.rename(str.lower,axis='columns',inplace=True)
KN_error.index = KN_error.index  - true_t0

plt.plot(KN_obs,'o')
plt.gca().invert_yaxis()

#taking earliest observation as t0
KN_obs.index = KN_obs.index - KN_obs.index[0]
KN_error.index = KN_error.index - KN_obs.index[0]

with pd.option_context('mode.use_inf_as_null', True):
    KN_obs = KN_obs.dropna(how='all')
KN_error = KN_error.dropna(how='all')
KN_error = KN_error
KN_obs.index = pd.to_timedelta(KN_obs.index,unit='d')
KN_error.index = pd.to_timedelta(KN_error.index,unit='d')

t0_start = 1.
t0_start = KN_obs.index[10].total_seconds()  / (3600 * 24)
KN_obs.drop(KN_obs.index[0:10],inplace=True)
KN_error.drop(KN_error.index[0:10],inplace=True)


KN_obs.index = KN_obs.index - KN_obs.index[0]
KN_error.index = KN_error.index - KN_obs.index[0]
LC_timeindex = KN_obs.index.total_seconds() / (3600*24)

time_index = {}
observations = {}
for x in filters:
    observations[x] = KN_obs[x].values
    observations[x] = KN_obs[x].values
    time_index[x] = np.array(LC_timeindex[~np.isnan(observations[x])])
    observations[x] = np.array(observations[x][~np.isnan(observations[x])])
Truth = [t_0, 44, 0.04,0.15,1.5,0.025,0.3,4.0]

def like_single(args):
    t0, DL, mej, vej, Xlan, mej_b, vej_b, Xlan_b = args
    filter_sum_prob = {}
    KN_modeltest = {}
    for x in filters:
        index = time_index[x] + t0
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
"""
priors['D_L'] = bilby.prior.Uniform(minimum=35,maximum=49,name='D_L')
priors['t0'] = bilby.prior.Uniform(minimum=0.,maximum=5,name='t0')
priors['mej'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej')
priors['vej'] = bilby.prior.Uniform(minimum=0.05,maximum=0.35,name='vej')
priors['Xlan'] = bilby.prior.Uniform(minimum=1.,maximum=2.,name='Xlan')
priors['mej_b'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej_b')
priors['vej_b'] = bilby.prior.Uniform(minimum=0.05,maximum=0.35,name='vej_b')
priors['Xlan_b'] = bilby.prior.Uniform(minimum=2.,maximum=9.,name='Xlan_b')
priors['vfrac'] = bilby.prior.Constraint(minimum=0.,maximum=0.25,name='vfrac')

"""
priors['D_L'] = bilby.prior.Uniform(minimum=43.,maximum=45.,name='D_L')
priors['t0'] = bilby.prior.Uniform(minimum=0.,maximum=3.,name='t0')
priors['mej'] = bilby.prior.Uniform(minimum=0.03,maximum=0.05,name='mej')
priors['vej'] = bilby.prior.Uniform(minimum=0.14,maximum=0.16,name='vej')
priors['Xlan'] = bilby.prior.Uniform(minimum=1.,maximum=2.,name='Xlan')
priors['mej_b'] = bilby.prior.Uniform(minimum=0.02,maximum=0.03,name='mej_b')
priors['vej_b'] = bilby.prior.Uniform(minimum=0.29,maximum=0.31,name='vej_b')
priors['Xlan_b'] = bilby.prior.Uniform(minimum=3.5,maximum=4.1,name='Xlan_b')
priors['vfrac'] = bilby.prior.Constraint(minimum=0.,maximum=0.25,name='vfrac')


print('About to run sampler')
print(datetime.now())

now = datetime.now()
l = '{}'.format(now.strftime("%Y-%m-%d-%H%M"))
rpath = 'Results/{}/{}'.format(run_type,l)

os.mkdir(rpath)
shutil.copy('calctdl_GPR_gw170817_tidy.py','Results/{}/{}/src_code.py'.format(run_type,l),follow_symlinks=True)


sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='dynesty', outdir='Results/{}/{}'.format(run_type,l),label=l,verbose = True, dlogz = .1, npoints = 2000, n_check_point = 300 )
#sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='ptemcee', outdir='Results/{}/{}'.format(run_type,l),label=l,nburn=Nburnin, nwalkers = Nens, ntemps=Ntemps,Tmax=Tmax,iterations=Nsteps)
#sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='dynesty', outdir='Results/{}/{}'.format(run_type,l),label=l,verbose = True, resume=True, dlogz = .01,npoints = 2000, n_check_point = 3000)

print('Run complete')
print(datetime.now())

np.savetxt('Results/{}/{}/posterior'.format(run_type,l),np.array(sampler.posterior))

sampler.plot_corner(truth={'t0':t0_start,'D_L':44,'mej':0.040,'vej':0.15,'Xlan':1.5,'mej_b':0.025,'vej_b':0.3,'Xlan_b':4.})

#sampler.plot_corner(truth={'mej':0.040,'vej':0.15,'Xlan':1.5,'mej_b':0.025,'vej_b':0.3,'Xlan_b':4.})


#sampler.plot_corner(truth={'t0':t0,'D_L':44,'mej':0.040,'vej':0.15,'Xlan':1.5,'mej_b':0.025,'vej_b':0.3,'Xlan_b':4.})

