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

# mix two datasets taken with different istance between the skull and the beads
# load the two datasets, remove the first 


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


rootfolder = '/home/moro/localdata/23032019/'
datasources = [ rootfolder+'data_23032019_all.mat']


commonseed = int(time.time()*10E4*np.random.rand())
dumpfolder = '/'.join(rootfolder.split('/')[:-2])+os.sep+str(commonseed)+os.sep
if not os.path.exists(dumpfolder):
    print(f'creating dir {dumpfolder}')
    os.makedirs(dumpfolder)

parset = []

# populate with different demixing parameters
for binss in [1, 20]:
    for lowfilter in [400]:
        for highfilter in [1.5]:
            dpars = copy.copy(sdmx.demixing_pars)
            dpars['binning'] = binss
            dpars['lowfilter'] = lowfilter
            dpars['highfilter'] = highfilter
            parset.append(dpars)

for fff in datasources:
    for dpars in parset:
        print('===================')
        demthread(fff, commonseed, dumpfolder, dpars)
