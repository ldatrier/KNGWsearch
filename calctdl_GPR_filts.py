import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
import os,glob
import pandas as pd
import random
import corner
from datetime import datetime
import bilby
from scipy.misc import logsumexp
import pickle

np.warnings.filterwarnings('ignore')

c = 3.0*10**8 #speed of light
h0 = 70.0 #Hubble constant
Lbolsun  = 3.0128*10**26 #Bolometric luminosity of the Sun in Watts
kmin_i = -16.7449619905
kmax_i = 15.469444374279998  #Constants for LC normalisation
kmin_r = -17.0326488143
kmax_r = 14.60078505241  #Constants for LC normalisation

F = 'GPR'

dl = 40. #DISTANCE TO SOURCE
Ntemps = 2 #Number of temperatures for ptemcee chains
Tmax = 2000 #Max temperature for hot PT chain
filters = ['i','r']
cadence = 1
startt = 1
Nburnin = 60   # number of burn-in samples
Nens = 150  # number of ensemble points
Nsteps = 1000 # number of steps for MCMC

ID = np.random.randint(0,100000)

gp_i=pickle.load(open('/home/laurencedatrier/Documents/PhD/Mergert_git/GPR_LC/gpr_model_i.sav','rb'))
gp_r=pickle.load(open('/home/laurencedatrier/Documents/PhD/Mergert_git/GPR_LC/gpr_model_r.sav','rb'))

def app_m(D,M,h):
    return M-5+5*np.log10(D*10**6)

m5 = pd.DataFrame({'u':[23.9,0.037],'g':[25.0,0.038],'r':[24.7,0.039],'i':[24.0,0.039],'z':[23.3,0.040],'y':[22.1,0.040]},index=['m5','gamma'],columns=['u','g','r','i','z','y'])
m5 = m5[filters]

def new_depth(N_visits,N_bands):
    new_m5 = m5[:'m5'] + 2.5 * np.log10(np.sqrt(N_visits/N_bands))
    return new_m5

def magerror(mags,filter): #function for LSST magnitude error (From LSST science book)
    x = 10**(0.4*(mags-m5[filter]['m5']))
    sig =  (0.04-m5[filter]['gamma'])*x + m5[filter]['gamma']*x**2
    errs = np.sqrt(sig + 0.003**2)
    mask = mags > m5[filter]['m5']
    errs[mask] = 0.01
    return errs

def LC_model(mejl,vejl,Xlan,time_index):
    x_test = np.array([[np.log10(mejl),np.log10(vejl),Xlan,time_index[y]] for y,t in enumerate(time_index)])
    LC_pred_i, sigma_i = gp_i.predict(x_test,return_std = True)
    LC_pred_r, sigma_r = gp_r.predict(x_test,return_std = True) #Rewrite this to adapt to all filters
    return (LC_pred_i*kmax_i) + kmin_i, sigma_i, (LC_pred_r*kmax_r) + kmin_r, sigma_r


testindex = np.arange(startt,10,cadence)
x_test = np.array([[np.log10(0.025),np.log10(0.3),4.,testindex[y]] for y,t in enumerate(testindex)])
y_pred_i, sigma_i = gp_i.predict(x_test,return_std=True)
KNLC_i = y_pred_i * kmax_i + kmin_i
sigmas_i =  sigma_i * kmax_i
LC_observations_i = app_m(dl,KNLC_i,h0)
y_pred_r, sigma_r = gp_r.predict(x_test,return_std=True)
KNLC_r = y_pred_r * kmax_r + kmin_r
sigmas_r =  sigma_r * kmax_r
LC_observations_r = app_m(dl,KNLC_r,h0)
LC_timeindex = testindex - startt
plt.plot(LC_observations_r,'o')
plt.plot(LC_observations_i,'o')

plt.gca().invert_yaxis()
plt.show()

def like_single(args):
    D_L, t0, mej, vej, Xlan = args
    index_gpr = LC_timeindex + t0
    x_test = np.array([[np.log10(mej),np.log10(vej),Xlan,index_gpr[y]] for y,t in enumerate(index_gpr)])
    y_pred_i, sigma_i = gp_i.predict(x_test,return_std=True)
    KNLC_i = y_pred_i * kmax_i + kmin_i
    sigmas_i =  sigma_i * kmax_i
    KN_modeltest_i = app_m(D_L,KNLC_i,h0)
    y_pred_r, sigma_r = gp_r.predict(x_test,return_std=True)
    KNLC_r = y_pred_r * kmax_r + kmin_r
    sigmas_r =  sigma_r * kmax_r
    KN_modeltest_r = app_m(D_L,KNLC_r,h0) #Double check the returned prob against maths
    return logsumexp((norm.logpdf(KN_modeltest_i,loc=LC_observations_i,scale=sigmas_i), norm.logpdf(KN_modeltest_r,loc=LC_observations_r,scale=sigmas_r)),axis=0)
import multiprocessing as mp
class KNmodelprob(bilby.Likelihood):
    def __init__(self, nthreads=None):

        self.parameters = {'t0':None,'mej': None, 'vej': None, 'Xlan': None}#{'t0': None, 'D_L': None, 'mej': None, 'vej': None, 'Xlan': None}

	self.pool = mp.Pool(nthreads)

    def log_likelihood(self):
        #D_L = self.parameters['D_L']
        #t0 = 1.
        D_L = 40.
        t0 = self.parameters['t0']
        mej = self.parameters['mej']
        vej = self.parameters['vej']
        Xlan = self.parameters['Xlan']
        args = [(D_L, t0, mej, vej, Xlan)]
	prob = self.pool.map(like_single, args)
        return np.sum(prob)


ndims = 4#5

priors = bilby.core.prior.PriorDict()
#priors['D_L'] = bilby.prior.PowerLaw(alpha=2,minimum=30,maximum=60,name='D_L')
#priors['D_L'] = bilby.prior.Uniform(minimum=30,maximum=45,name='D_L')
priors['t0'] = bilby.prior.Uniform(minimum=0.5,maximum=4,name='t0')
priors['mej'] = bilby.prior.Uniform(minimum=0.001,maximum=0.1,name='mej')
priors['vej'] = bilby.prior.Uniform(minimum=0.05,maximum=0.5,name='vej')
priors['Xlan'] = bilby.prior.Uniform(minimum=1.5,maximum=9,name='Xlan')


print('About to run sampler')
print(datetime.now())

now = datetime.now()
l = '{}_{}'.format(now.strftime("%Y-%m-%d"),ID)
rpath = 'Results/{}/{}'.format(F,l)

os.mkdir(rpath)
#sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='ptemcee', outdir='Results/{}/{}'.format(F,l),label=l,nburn=Nburnin, nwalkers = Nens, ntemps=Ntemps,Tmax=Tmax,iterations=Nsteps)
sampler = bilby.run_sampler(KNmodelprob(), priors, sampler='dynesty', outdir='Results/{}/{}'.format(F,l),label=l,verbose = True, dlogz = 0.2,npoints = 2000)


plt.figure(figsize=(10,10))
plt.gca().invert_yaxis()
plt.plot(LC_observations_r,'.--')
plt.plot(LC_observations_i,'.--')

plt.savefig('Results/{}/{}/{}_{}OBS.png'.format(F,l,now.strftime("%Y-%m-%d"),ID))

information = open("Results/{}/{}/{}_{}information.txt".format(F,l,now.strftime("%Y-%m-%d"),ID),"w")
information.write("True d_L (Mpc) {}".format(dl))
information.write("True t0 (days){}".format(startt))
information.write("Running over models:{}".format(F))
information.write("Number of temps:{}".format(Ntemps))
information.write("Max Temp:{}".format(Tmax))
information.write("Number of samples:{}".format(Nens))
information.write("Burn-in samples:{}".format(Nburnin))
information.write("Number of steps:{}".format(Nsteps))
information.write("Filters used:{}".format(filters))
information.write("Cadences for ugrizY:{}".format(cadence))
information.write("Start of obs. for ugrizY:{}".format(startt))
information.close()

print('Run complete')
print(datetime.now())


np.savetxt('Results/{}/{}/{}_{}posterior'.format(F,l,now.strftime("%Y-%m-%d"),ID),np.array(sampler.posterior))

#sampler.posterior=sampler.posterior.rename(columns={'D_L':'$d_L$ (Mpc)','t0':'$t_0$ (days)','mej':'mej','vej':'vej','Xlan':'Xlan'})
sampler.plot_corner(truth={'t0':1.,'mej':0.025,'vej':0.3,'Xlan':4.})
#sampler.plot_corner(truth={'$t_0$ (days)':startt,'$d_L$ (Mpc)':dl,'mej':0.025,'vej':0.3,'Xlan':4.})
sampler.plot_walkers()
