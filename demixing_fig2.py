import sys
import speckle_stuffs as spkstf
import speckled_neurons_demixing as sdmx
from skimage.io import imread
import numpy as np
import pickle
from pytictoc import TicToc
import time
import copy
import argparse
import os
import scipy
import time


t = TicToc()

def demthread(datasource, commonseed, dumpfolder, dpars):
    print(f'{datasource}')
    t0 = time.time()
    video_real = np.swapaxes(scipy.io.loadmat(datasource)['video_data'], 0, 2)
    dpars['dataset'] = datasource
    starttime = int(time.time()*10E5*np.random.rand())
    print(starttime)
    video_real = video_real[-500:]
    [traces, spatial, err_progress] = sdmx.demix_speckles_video(video_real, dpars)

    picklename = datasource.split('.')[0]+'_%d_%d.pickle'%(commonseed, starttime)
    picklename = dumpfolder+picklename.split('/')[-1]
    filehandler = open(picklename, 'wb')
    dpars['computing time'] = time.time()-t0
    dpars['error_progression'] = err_progress
    pickle.dump([traces, spatial, dpars], filehandler, pickle.HIGHEST_PROTOCOL)
    filehandler.close()
    print(f'Written to {picklename}')


rootfolder = '/home/moro/localdata/08032019/'
datasetrange = [1, 2, 3]
datasources = [ rootfolder+ff for ff in os.listdir(rootfolder)\
               if (ff[-4:] == '.mat') and (len(ff) == 21) and 
               ff[-7:] in [f'{str(a).zfill(3)}.mat' for a in datasetrange]
               ]

commonseed = int(time.time()*10E4*np.random.rand())
dumpfolder = '/'.join(rootfolder.split('/')[:-2])+os.sep+str(commonseed)+os.sep
if not os.path.exists(dumpfolder):
    print(f'creating dir {dumpfolder}')
    os.makedirs(dumpfolder)

parset = []

# perform the analisys for multiple binning
for binss in [5, 10, 20, 30, 40, 50, 100]:
    dpars = copy.copy(sdmx.demixing_pars)
    dpars['binning'] = binss
    parset.append(dpars)

for fff in datasources:
    for dpars in parset:
        print('===================')
        demthread(fff, commonseed, dumpfolder, dpars)
