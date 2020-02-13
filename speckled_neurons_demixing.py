"""
here the demixing, not the coupling and the evaluation with the groundtruth dataset
only the video must be taken as input, and the different parameters

here are included also the support functions, and the display functions, the coupliungs, and similars
(which then will be called by the external script)
the code if runned standalone must just analize the video, and return a csv with the traces and a multipage
tiff with the components
"""


import matplotlib
# if there is no exported X, this can be seteed on
# matplotlib.use('Agg')
import sys
# sp_NMF is a custom version of classic sklearn implementation, just with dump of errors over iterations
from sklearn.decomposition import sp_NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import argparse
import os
import numpy as np
import time

from pytictoc import TicToc
from skimage.io import imread
import scipy.io as sio
import speckle_stuffs as spkstf

from sys import getsizeof


demixing_pars = {
    'alpha' : 0, 
    'l1overl2' : 0,
    'solver' : 'cd', # 'ma'
    'max_iter' : 200,
    'beta_loss' : 'frobenius', # 'itakura-saito' 'kullback-leibler', latter only with 'mu' solver
    'components' : 1,
    'lowfilter' : 600, 
    'highfilter' : 1.5,
    'binning' : 20,
    'bit depth' : 16,
    'computing time' : 0,
    'init' : 'nndsvd',
    'error_progression' : [],
    'scree_plot' : [],
    'double initialization' : False,
    'dataset' : '',
    'frames_no' : 0,
    'gt_frame_start' : 0
}

def demix_speckles_video(video, demixing_pars=demixing_pars, tips=None, return_error=False):
    # tips could be stuffs like the numebr of neurons, the speckle size, or 
    # other paramenters which should in a letter stage understanded by the algorithm itself   
    t = TicToc()
    for kk in demixing_pars.keys():
        print(kk, demixing_pars[kk])

    print('searching for number of components...')
    # perform SVD and try to estimate the number of components in the system
    # subsampling
    video_tmp = spkstf.bin_video(video, demixing_pars['binning'])
    frames_no = video_tmp.shape[0]
    xsize = video_tmp.shape[1]
    ysize = video_tmp.shape[2]
    X = video_tmp.reshape((frames_no, xsize*ysize))
    del video_tmp

    # SVD
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    svd.fit(X)  
    # remove the first larger component
    tmpv = np.log(svd.singular_values_[1:])
    
    demixing_pars['scree_plot'] = svd.singular_values_
    kmeans = KMeans(n_clusters=2).fit(tmpv.reshape((-1, 1)))
    labels = kmeans.labels_
    print('components no updated ', str(demixing_pars['components']), '->', str(sum(labels==labels[0])))
    demixing_pars['components'] = sum(labels==labels[0])

    print('spatial filtering the dataset...')
    t.tic()
    video =  spkstf.gauss_don_filt_GPU(video, lp=demixing_pars['lowfilter'], hp=demixing_pars['highfilter'])
    t.toc()

    print('binning...')
    video = spkstf.bin_video(video, demixing_pars['binning'])
    frames_no = video.shape[0]
    xsize = video.shape[1]
    ysize = video.shape[2]

    print('start decomposition...')
    t.tic()
    
    # one more component for the background
    X = video.reshape((frames_no, xsize*ysize))
    del video
    nmfsvd = sp_NMF(
        n_components=demixing_pars['components']+1, 
        init=demixing_pars['init'], random_state=0,\
        alpha=demixing_pars['alpha'], l1_ratio=demixing_pars['l1overl2'], 
        solver=demixing_pars['solver'], max_iter=demixing_pars['max_iter'], beta_loss=demixing_pars['beta_loss'], 
        return_error=return_error
        )
    
    W = nmfsvd.fit_transform(X)
    H = nmfsvd.components_
    error_progression = nmfsvd.error_progression

    traces = W.T
    # return the reshaped matrix with spatial footprints
    footprints = H.reshape((demixing_pars['components']+1, xsize, ysize))

    t.toc()
    print('decomposition done.')

    # W traces
    # H spatial footprints
    return [traces, footprints, error_progression]


if __name__=="__main__":
    # parse argv and Co
    # the output can be csv, or also a pickle? or something else
    video = imread(datasource)
    [Ws, Hs] = demix_speckles_video(video, demixing_pars)

    grtrh_traces = [nn.trace for nn in neurons]
    extr_traces = [Ws[:,cmp] for cmp in np.arange(neuronno, -1, -1)]
    
    grtrh = sio.loadmat(gtsource)
    grtrh_traces = grtrh['pat']

    couplings = splst.find_trace_couplings(grtrh_traces, extr_traces, neuronno)

    ccorrelations = [dd[2] for dd in couplings]
    meancp = np.mean(ccorrelations)
    semcp = np.std(ccorrelations)/np.sqrt(len(ccorrelations))