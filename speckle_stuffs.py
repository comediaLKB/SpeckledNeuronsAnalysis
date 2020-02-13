# chech if everything load is used..probably not...

import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn.decomposition as skdec

import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from skimage.io import imread
import skimage.morphology as mrph
from skimage.filters import threshold_otsu
from multiprocessing import Pool

from moro_utils import printProgressBar, eprint


def bin_video(video, binning):
    """bin_video(matrix[<X> x <Y> x <frames>], binning) -> matrix[<X/binning> x <Y/binning> x <frames>]
    bin the frames of a video (averaged across <binninb> pixels"""
    return np.array([np.mean([
        frame[
            ox:-binning-np.shape(video)[1]%binning:binning, 
            oy:-binning-np.shape(video)[2]%binning:binning
        ]
        for ox in range(binning) 
        for oy in range(binning)
    ], axis=0) for frame in video]).astype(np.int16)


def crosscorr(img1, img2, use_GPU=False):
    """ crosscorr(img1, img2) -> img3
    perform crosscorrelation between img1 and img2 """
    if use_GPU:
        crosscorr_GPU(img1, img2)
    fft_product = (np.fft.fft2(img1) * np.fft.fft2(img2).conj())
    cc_data0 = np.abs(np.fft.fftshift(np.fft.ifft2(fft_product)))
    return cc_data0

def crosscorr_GPU(img1, img2):
    raise GeneralWarning('not implemented yet')
    ### To be implementd
    if np.shape(img1) != np.shape(img2):
        raise ValueError("Error: images must be the same size")
        return 0
    n1, n2 = img1.shape

    try:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.fft as cu_fft
        import skcuda.linalg as linalg
    except ImportError:
        eprint('cannot import cuda libraries, crosscorrelating with CPU...')
        return crosscorr(img1, img2)

    # dot product of the fft and the conj fft
    # ifft
    # fftshift
    # abs

    # send the images to the GPU
    img1gpu = gpuarray.to_gpu(img1.astype('complex64')) #alternativamente float32
    img2gpu = gpuarray.to_gpu(img2.astype('complex64'))
    # cc_gpu = gpuarray.empty((2, n1, n2), dtype=np.complex128)

    plan_fft = cu_fft.Plan(img1.shape, np.complex64, np.complex64, batch=2)
    plan_ifft = cu_fft.Plan(img1.shape, np.complex64, np.complex64, batch=2)

    cu_fft.fft(img1gpu, img1gpu, plan_fft) # try to keep in the same memory slot, works?
    cu_fft.fft(img2gpu, img2gpu, plan_fft)
    img2gpu = linalg.conj(img2gpu)
    cc_gpu = linalg.dot(img1gpu, img2gpu)
    cu_fft.fft(cc_gpu, cc_gpu, plan_ifft)

    crosscorr = cc_gpu.get()

    # delete some stuff
    del(img1gpu)
    del(img2gpu)
    del(cc_gpu)

    return crosscorr
    ### to be implemented


def zero_norm_crosscorr(img1, img2):
    """ zero_norm_crosscorr(img1, img2) -> type(img1)
    zero norm crosscorrelation (with zero shift)"""
    return (1./np.size(img1)) * np.sum( 
        (1./(np.std(img1)*np.std(img2)))*(img1-np.mean(img1))*(img2-np.mean(img2)) 
        )

def norm_crosscorr(img1, img2):
    """ norm_crosscorr(img1, img2) -> type(img1)
    normalized cross correlation with zero shift"""
    return (1./np.size(img1)) * np.sum( 
        (1./(np.std(img1)*np.std(img2)))*(img1)*(img2) 
        )

def pupil(matrix, diameter=1):
    """ pupil(matrix, radius_px=1) -> type(matrix)
    overlap a circular aperture to the matrix. diameter define the diameter of the pupil as ratio with the minor axis of the matrix """
    xdim, ydim = matrix.shape[:2]
    radius_px = (np.min(matrix.shape[:2])/2.)*diameter
    YYY, XXX = np.meshgrid(np.arange(ydim)-ydim/2.+0.5, np.arange(xdim)-xdim/2.+0.5)
    pupil = np.sqrt(XXX**2+YYY**2) < radius_px
    return np.multiply(matrix, pupil)

def draw_a_disk(base_img, center, r_spot, intensity=1):
    """ draw_a_disk(base_img, center, r_spot, intensity=1) -> type(base_img)
    put a circle of radius "r_spot" in a certain position "center", with given intensity """
    (x, y) = center
    xm, ym = np.shape(base_img)
    xm = int(xm/2.)
    ym = int(ym/2.)
    xxx, yyy = np.mgrid[-xm:xm, -ym:ym]
    xxx = xxx-x
    yyy = yyy-y
    diskimag = ((xxx**2 + yyy**2) < r_spot**2)*intensity
    return np.add(base_img, diskimag)

def draw_a_point(base_img, center, intensity=1):
    """ draw_a_point(base_img, center, intensity=1) -> type(base_img)
    put a point of the emitted neuron in the position "center", with given intensity """
    (x, y) = np.uint16(center)
    xm, ym = np.shape(base_img)
    xm = np.uint16(xm/2.)
    ym = np.uint16(ym/2.)
    base_img[xm-x, ym-y] = intensity
    return base_img

def place_rnd_pt_in_fov(rfov, timeout=500):
    for i in range(timeout):
        pt = np.random.rand(2)-0.5
        if np.sqrt(np.sum(pt**2))<0.5:
            break
    return (pt[0]*rfov*2, pt[1]*rfov*2)

def place_rnd_pts_in_fov(ptno, rfov, mindist=0, timeout=500):
    # place the first point
    pts = np.array([place_rnd_pt_in_fov(rfov)])
    # find a place for the other points
    for ptidx in range(1,ptno):
        for i in range(timeout):
            pt = place_rnd_pt_in_fov(rfov)
            if not np.min([np.sqrt(np.sum((np.subtract(pt, pp))**2)) for pp in pts])<mindist:
                # print(np.min([np.sqrt(np.sum((np.subtract(pt, pp))**2)) for pp in pts]))
                break
        if (i+1)==timeout:
            eprint('timeout reached, placing a point with no min distance constrain...')
            pts = np.append(pts, [pt], axis=0)
        pts = np.append(pts, [pt], axis=0)
    return pts

def contrast(rawdata, decimate='None', method='max'):
    """ contrast(rawdata, decimate='None', method='max') -> float
    Ca.lculate the contrast in different ways.
    If no parameters are setted, just std/mean.
    Decimate will evaluate the contrast and give back the max found, or the mean of the values, depenging on "method".
    The decimate value is the decimation (not the step). 
    ratio, so dstep will be the step """

    if decimate=='None':
        return np.std(rawdata)/np.mean(rawdata)

    [xsize, ysize] = np.shape(rawdata)
    dstep = np.int16(np.min([xsize, ysize])/decimate) 
    tmpcontrt = []
    for xx in np.arange(0, xsize, dstep):
        for yy in np.arange(0, ysize, dstep):
            tmp = rawdata[xx:xx+dstep, yy:yy+dstep]
            tmpcontrt = np.append(tmpcontrt, np.std(tmp)/np.mean(tmp))

    if method=='max':
        return np.max(tmpcontrt)
    elif method=='mean':
        return np.mean(tmpcontrt)


def pearson_crosscorr(t1, t2):
    """ pearson croscorrelation between two traces/array"""
    t1 = np.squeeze((t1 - np.min(t1))/(np.max(t1)-np.min(t1)))
    t2 = np.squeeze((t2 - np.min(t2))/(np.max(t2)-np.min(t2)))
    return ((np.correlate(t1, t2)[0])**2/(np.correlate(t1, t1)[0]*np.correlate(t2, t2)[0]))


def trace_correlation(t1, t2, method='zncc'):
    """ define the trace correlation"""
    if method=='zncc':
        # zero norm crosscorrelation
        return zero_norm_crosscorr(t1, t2)
    elif method=='pcc':
        # pearson cross correlation
        return pearson_crosscorr(t1, t2)


def find_trace_couplings(groundtr, extractr, timeout=100, method='zncc'):#, neuronno):
    # method can be zncc, pcc, ncc
    # check which groundtruth resemble better
    # then do it again. store all the cross correlation score.
    # arrive at the end, and try to maximize the overall score, leaving the first component alone
    # if a trace is the best in two cases, then the latter trace found have the couple set to np.NaN
    # so there will be an error and I will solve the code with these dataset
    couplings = []
    for i, grt in enumerate(groundtr):
        correlations = []
        for j, tr in enumerate(extractr):
            correlations = np.append(correlations, trace_correlation(tr, grt, method=method))
        couplings.append([i, np.argmax(correlations), np.max(correlations)])
	# take a look if there is something with multiple match, and in case try to find a solution
    for ttt in range(timeout):
        # put the ground truth index and matched traces index in two arrays for comodity
        ground_idx = np.asarray([c[0] for c in couplings])
        matched_extraces = np.asarray([c[1] for c in couplings])
        # uniques have in the first column the unique indexes, and in the second the count (amount of time that they apprear)
        uniques = np.unique(matched_extraces, return_counts=True)
        # check if there is something to rematch, and in case put in "multiple_match" the ones which need rematching
        multiple_match = uniques[0][uniques[1]>1]
        # if all has just one match, then skip the cycle
        if not np.any(multiple_match):
            break
        # cicle over the traces which are not unique
        for mm in multiple_match:
            to_be_rematched = ground_idx[[idx == mm for idx in matched_extraces]]
            correlations = []
            for idx in to_be_rematched:
                correlations = np.append(correlations,\
							trace_correlation(groundtr[idx], extractr[mm], method=method))
            bestmatch_idx = np.argmax(correlations)
            to_be_rematched = np.delete(to_be_rematched, bestmatch_idx)

        # match with the missing ones
        missing = ground_idx[[mm not in matched_extraces for mm in ground_idx]]
        # chech if there are still some exrated traces to match with the missing ones
        # if the multiple matched correspond to the fact that we dont have enough exracted traces, remove the extra and avoid to reiterate the cycle
        if len(multiple_match) == (len(groundtr)-len(extractr)):
            # remove the extras from couplings
            for mm in multiple_match:
                # keep only the one with the greatest correlation
                couplings_to_check = [cc for cc in couplings if cc[1]==mm]
                idx_of_couplings_to_check = [iij for (iij, cc) in enumerate(couplings) if cc[1]==mm]
                # in fluent python there should be a clean way to do so. i am in rush and I don't remember. ill do an horrible stuff (ans already happened...)
                to_keep = np.argmax([cc[2] for cc in couplings_to_check])

                idx_of_couplings_to_check.pop(to_keep)
                for tr in idx_of_couplings_to_check:
                    couplings.pop(tr)
                to_remove = [cc for cc in couplings_to_check if cc[0]!=couplings_to_check[to_keep][0]]
            break

        # change the entries in "to_be_rematched" to the ones which have the best correlations with the missing
        for idx in to_be_rematched:
            correlations = []
            for mm in missing:
                try: # this try should not be needed...to check
                    correlations = np.append(correlations, trace_correlation(groundtr[idx], extractr[mm], method=method))
                    bestmatch_idx = missing[np.argmax(correlations)]
                    couplings[idx] = [idx, bestmatch_idx, np.max(correlations)]
                except IndexError:
                    pass
            # after some error I moved these into the "try"
            # bestmatch_idx = missing[np.argmax(correlations)]
            # couplings[idx] = [idx, bestmatch_idx, np.max(correlations)]
    # could be that the multiple match count correspond to the difference len(extractr)-len(groundtr)
    # take the difference len(extractr)-len(groundtr) and remove the ones with the lowest correlations, which are more likely the spurious ones
    # there is still multiple match? if yes, check if removing them will match the number of extrated traces (which could be less than the used traces)

    return couplings

def print_dic(dic, plot_arrays=True):
    for kk in dic.keys():
        # if np.shape(dic[kk])>3:
        if (type(dic[kk]) is list):
            if(type(dic[kk][0]) is [float, int]): 
                if plot_arrays:            
                    import matplotlib.pyplot as plt
                    plt.plot(dic[kk])
                    plt.show()
            else:
                break
        else:
            print('%s\t%s'%(kk, str(dic[kk])))
    pass

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def gaussian_profile(shape, center, width):
    (x, y) = center
    [x, y] = [x+shape[0]/2, y+shape[1]/2]
    return gaussian(1, x, y, width, width)(*np.indices(shape))

def draw_a_gaussian(img, center, width, intensity=1):
    shape = np.shape(img)
    (x, y) = center
    [x, y] = [x+shape[0]/2, y+shape[1]/2]
    return np.add(img, gaussian(intensity, x, y, width, width)(*np.indices(shape)))


def gaussian_donut(shape, inner_width, outer_width):
    if outer_width==0:
        return -gaussian(1, shape[0]/2, shape[1]/2, inner_width, inner_width)(*np.indices(shape))
    elif inner_width==0:
        return gaussian(1, shape[0]/2, shape[1]/2, outer_width, outer_width)(*np.indices(shape))
    else:
        return gaussian(1, shape[0]/2, shape[1]/2, outer_width, outer_width)(*np.indices(shape))-\
                        gaussian(1, shape[0]/2, shape[1]/2, inner_width, inner_width)(*np.indices(shape))
    
def gauss_don_filt(matrix, hp = 0, lp = np.inf):
    # lp is the lowpass filer cutoff
    # hp is the highpass filter cutoff 
    # must be called only once the gaussian_donut
    # so if is a video or a single frame, understand it, and do it once
    if len(np.shape(matrix))>2:
        if np.isinf(lp):
            lp = 2*np.max(np.shape(matrix)[1:3])
        fftfiltermask = gaussian_donut(np.shape(matrix)[1:3], hp, lp)
        filtered = np.array([
            np.abs(np.fft.ifft2(np.multiply(np.fft.fft2(frame), np.fft.fftshift(fftfiltermask))))
            for frame in matrix
            ])
        return filtered.astype(np.int16)

    elif len(np.shape(matrix))==2:
        if np.isinf(lp):
            lp = 2*np.max(np.shape(matrix))

        fftfiltermask = gaussian_donut(matrix.shape, hp, lp)
        filt_fourier = np.multiply(np.fft.fft2(matrix), np.fft.fftshift(fftfiltermask))
        return np.abs(np.fft.ifft2(filt_fourier)).astype(np.int16)

def gauss_don_filt_GPU(video, hp = 0, lp = np.inf):
    # lp is the lowpass filer cutoff
    # hp is the highpass filter cutoff 
    # must be called only once the gaussian_donut
    # so if is a video or a single frame, understand it, and do it once

    try:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.fft as cu_fft
    except ImportError:
        eprint('cannot import cuda libraries, filtering with the CPU...')
        return gauss_don_filt(video, lp=lp, hp=hp)

    if len(np.shape(video))>2:
        if np.isinf(lp):
            lp = 2*np.max(np.shape(video)[1:3])
        mask = gaussian_donut(np.shape(video)[1:3], hp, lp)

        n1, n2 = mask.shape
        mask = mask.astype('complex64')

        # prepare the plans
        plan_forward = cu_fft.Plan((n1, n2), np.float32, np.complex64)
        plan_backward = cu_fft.Plan((n1, n2), np.complex64, np.float32)
        # preallocate the filtered video
        filtvideo = np.zeros_like(video)

        for idx,frame in enumerate(video):
            printProgressBar(idx, np.shape(video)[0])
            # Convert the input array to single precision float
            frame = frame.astype('float32')
            # From numpy array to GPUarray
            framegpu = gpuarray.to_gpu(frame)

            # Initialise output GPUarrays
            fftframegpu = gpuarray.empty((n1,n2//2 + 1), np.complex64)
            filteredframegpu = gpuarray.empty((n1,n2), np.float32)
            # Forward FFT
            cu_fft.fft(framegpu, fftframegpu, plan_forward)
            # filter the FFT
    #         linalg.multiply(maskgpu, fftframegpu, overwrite=True)
            ####### here going back and forth with the GPU ram since something does not work with thje nvcc compiler...
            left = fftframegpu.get()
            if n2//2 == n2/2:
                right = np.roll(np.fliplr(np.flipud(fftframegpu.get()))[:,1:-1],1,axis=0)
            else:
                right = np.roll(np.fliplr(np.flipud(fftframegpu.get()))[:,:-1],1,axis=0) 
            fftframe = np.hstack((left,right)).astype('complex64')
            #### 
            fftframe = np.multiply(np.fft.fftshift(mask), fftframe).astype('complex64')
            # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
            fftframe = np.asarray(fftframe[:,0:n2//2 + 1], np.complex64)
            #### returin back to the GPU
            fftframegpu = gpuarray.to_gpu(fftframe) 
            
            # Backward FFT
            cu_fft.ifft(fftframegpu, filteredframegpu, plan_backward)
            filtvideo[idx] = np.abs(filteredframegpu.get()/n1/n2)

        return filtvideo.astype(np.int16)
    else:
        # only one frame; exactly what is done above, but with just one frame
        frame = video
        if np.isinf(lp):
            lp = 2*np.max(np.shape(frame))
        mask = gaussian_donut(np.shape(frame), hp, lp)

        n1, n2 = mask.shape
        mask = mask.astype('complex64')

        # prepare the plans
        plan_forward = cu_fft.Plan((n1, n2), np.float32, np.complex64)
        plan_backward = cu_fft.Plan((n1, n2), np.complex64, np.float32)
        
        # preallocate the filtered frame
        filtframe = np.zeros_like(frame)

        # Convert the input array to single precision float
        frame = frame.astype('float32')
        # From numpy array to GPUarray
        framegpu = gpuarray.to_gpu(frame)

        # Initialise output GPUarrays
        fftframegpu = gpuarray.empty((n1,n2//2 + 1), np.complex64)
        filteredframegpu = gpuarray.empty((n1,n2), np.float32)
        # Forward FFT
        cu_fft.fft(framegpu, fftframegpu, plan_forward)
        # filter the FFT
#         linalg.multiply(maskgpu, fftframegpu, overwrite=True)
        ####### here going back and forth with the GPU ram since something does not work with thje nvcc compiler...
        left = fftframegpu.get()
        if n2//2 == n2/2:
            right = np.roll(np.fliplr(np.flipud(fftframegpu.get()))[:,1:-1],1,axis=0)
        else:
            right = np.roll(np.fliplr(np.flipud(fftframegpu.get()))[:,:-1],1,axis=0) 
        fftframe = np.hstack((left,right)).astype('complex64')
        #### 
        fftframe = np.multiply(np.fft.fftshift(mask), fftframe).astype('complex64')
        # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
        fftframe = np.asarray(fftframe[:,0:n2//2 + 1], np.complex64)
        #### returin back to the GPU
        fftframegpu = gpuarray.to_gpu(fftframe) 
        
        # Backward FFT
        cu_fft.ifft(fftframegpu, filteredframegpu, plan_backward)
        filtframe = np.abs(filteredframegpu.get()/n1/n2)

        return filtframe.astype(np.int16)

        
def extract_traces_from_mat(matfile):
    import scipy.io as sio
    try:
        grtrh = sio.loadmat(matfile)
    except TypeError:
        print('error with {}'.format(matfile))
    # the mat lab variable 'pat' can be changed
    return grtrh['pat']

def extract_from_mat(matfile, var=None):
    """ from the file "matfile" extract the variable "var" (string) """
    try:
        import scipy.io as sio
        f = sio.loadmat(matfile)
    except NotImplementedError:
        import h5py
        f =  h5py.File(matfile, 'r')

    if var==None:
        print('Define which variable extract from .mat file.\nAvalable variable are (only first level):')
        print(list(f.keys()))
        return 0
    else:
        return f[var]
    # return matvar[var]

def plot_components_and_gt(extr_traces, extr_speckles, grtrh_traces, grtrh_speckles=None, couplings=None, outfile=None, show_thr=0):
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec

    components = np.shape(grtrh_traces)[0]
    # components = np.shape(extr_traces)[0]

    if (couplings is 'couple'):
        spkstf.eprint('search for couplings...')
        couplings = find_trace_couplings(grtrh_traces, extr_traces)
    elif (couplings is None):
        # use 1:1 couplings
        print('using 1:1 couplings...')
        couplings = [[idx, idx, 0] for idx in range(components)]

    colors = sns.color_palette("Set2", components)
    mpl.style.use('seaborn')
    trfig = plt.figure(figsize=(20,20))
    axgrid = gridspec.GridSpec(components*2, 20)

    for [idx, extridx, coup] in couplings:
        idx = int(idx)
        slot = idx*2
        extridx = int(extridx)
        if (grtrh_speckles is not None):
            # plot speckle ground truth
            plt.subplot(axgrid[ slot:slot+2 , 0:3])
            # cmin = np.mean(grtrh_speckles[idx])-3*np.std(grtrh_speckles[idx])
            # cmax = np.mean(grtrh_speckles[idx])+3*np.std(grtrh_speckles[idx])
            cmin = np.min(grtrh_speckles[idx])
            cmax = np.max(grtrh_speckles[idx])
            # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
            plt.imshow(grtrh_speckles[idx], cmap='Greys_r', clim=[cmin, cmax])
            plt.yticks([]), plt.xticks([])
        # plot found speckle
        plt.subplot(axgrid[ slot:slot+2 , 3:6])
        # cmin = np.mean(extr_speckles[extridx])-3*np.std(extr_speckles[extridx])
        # cmax = np.mean(extr_speckles[extridx])+3*np.std(extr_speckles[extridx])
        cmin = np.min(extr_speckles[extridx])
        cmax = np.max(extr_speckles[extridx])
        # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        plt.imshow(extr_speckles[extridx], cmap='Greys_r', clim=[cmin, cmax])
        plt.yticks([]), plt.xticks([])
        # plt.text(coupl)
        # plot ground truth temporal component

        
        plt.subplot(axgrid[ slot, 6:-1])
        plt.plot(grtrh_traces[idx], color=colors[idx])
        plt.yticks([]), plt.xticks([])
        # plot found temporal component
        ax = plt.subplot(axgrid[ slot+1, 6:-1])
        plt.plot(extr_traces[extridx], color=colors[idx])
        if coup < show_thr:
            ax.set_facecolor((1.0, 0.9, 0.9))
        plt.yticks([]), plt.xticks([])
    plt.tight_layout()
    if outfile is None:
        plt.show()
    else:
        trfig.savefig(outfile, format='pdf',  dpi=600)
        # trfig.savefig(outfile)


def plot_components(Ws, Hs):
    """
    if savefig is defined with a filename, then the resulting figure will be saved in a file
    """
    # an idea would be to read the shapes of Ws and Hs, and then 
    # iterate though them...is there is only one, ok, if there are two, then there will be 
    # a comparison; 3 comparison in different axes; 4 or more, comparison with the first in 
    # one axes, and all the others in the same axes

    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec

    print(np.shape(Hs)[0])
    components = int(np.shape(Hs)[0])
    colors = sns.color_palette("Set2", components)
    mpl.style.use('seaborn')
    trfig = plt.figure(figsize=(20,20))


    axgrid = gridspec.GridSpec(components, 20)

    for idx in range(components):
        plt.subplot( axgrid[ idx, 2:-1])
        plt.plot(Ws[:,idx], color=colors[idx])
        plt.yticks([]), plt.xticks([])

        plt.subplot(axgrid[ idx:idx , 0:1])
        cmin = np.mean(Hs[idx,:,:])-3*np.std(Hs[idx,:,:])
        cmax = np.mean(Hs[idx,:,:])+3*np.std(Hs[idx,:,:])
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        plt.imshow(Hs[idx,:,:], cmap=cmap, clim=[cmin, cmax])
        plt.yticks([]), plt.xticks([])
    plt.show()


def rebuild_video(Ws, Hs):
    """ rebuild the video from the components: video - Ws Hs
    Ws is the temporal activity
    Hs is the speckle patterns """
    # must be include some error check. as shape across the matrices
    # must be done with footprints and traces in input
    framesize = np.shape(Hs)[-1]
    frames = np.shape(Ws)[0]
    components = np.shape(Hs)[0]

    recontr_video = np.matmul(Ws, np.reshape(Hs,(components, framesize**2)))
    recontr_video = np.reshape(recontr_video, (frames, framesize, framesize))

    return recontr_video

def reconstruction_fidelity(Ws, Hs, video):
    def normalize(video):
        video = video - np.min(video)
        video = video/np.max(video)
        return video
    recontr_video = rebuild_video(Ws, Hs)
    # return np.sqrt(np.mean(np.square(normalize(recontr_video)-normalize(video))))
    return 1- np.std(normalize(recontr_video)-normalize(video))/np.mean(normalize(video))


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    Credits: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def take_half_cm(cmap):
    '''
    Modified from: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    for ri, si in zip(np.linspace(0, 1.0, 257), np.linspace(0, 1.0, 129)):
        r, g, b, a = cmap(ri+0.5)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = mpl.colors.LinearSegmentedColormap('halfcm', cdict)
    # plt.register_cmap(cmap=newcmap)
    return newcmap



def plot_correlations(ax1, ax2, fig, traces_cc, footprints_cc, method='zncc'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
    aspect = 20
    pad_fraction = 0.5

    if method == 'mixed':
        cmpp = take_half_cm(mpl.cm.BrBG)
        im1 = ax1.imshow(traces_cc, cmap=cmpp, interpolation = 'nearest', clim=[0,1])
    elif method == 'zncc':
        im1 = ax1.imshow(traces_cc, cmap='BrBG', interpolation = 'nearest', clim=[-1,1])


    divider = make_axes_locatable(ax1)
    width = axes_size.AxesY(ax1, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax1 = divider.append_axes("right", size=width, pad=pad)

    cbar = fig.colorbar(im1, ax=ax1, cax=cax1)
    if method == 'mixed':
            cbar.set_label('pearson cross-correlation')
    elif method == 'zncc':
        cbar.set_label('zero norm. cross-correlation')
    ax1.set_xlabel('extracted traces')
    ax1.set_ylabel('ground truth traces')
    ax1.set_title('traces')
    ax1.set_xticks([])
    ax1.set_yticks([])

#     im2 = ax2.imshow(footprints_cc, cmap='BuGn', clim=[0,1], interpolation = 'nearest')
    im2 = ax2.imshow(footprints_cc, cmap='BrBG', interpolation = 'nearest', clim=[-1,1])

    divider = make_axes_locatable(ax2)
    width = axes_size.AxesY(ax2, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax2 = divider.append_axes("right", size=width, pad=pad)

    cbar = fig.colorbar(im2, ax=ax2, cax=cax2)
    cbar.set_label('zero norm. cross-correlation')
    ax2.set_xlabel('extracted footprints')
    ax2.set_ylabel('ground truth footprints')
    ax2.set_title('footprints')
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()


def pvalue_stars(p):
    ss = ''
    if p>=0.05: return 'ns'
    if p<0.05:   ss +='*'
    if p<0.01:   ss +='*'
    if p<0.001:  ss +='*'
    if p<0.0001: ss +='*'
    return ss


def build_cc_mtrxs(grtrh_traces, extr_traces, grtrh_footprints, extr_footprints, method='zncc', halfmatrix=True):
    if method == 'zncc':
        couplings = np.array(find_trace_couplings(grtrh_traces, extr_traces, method='zncc'))
    elif method == 'mixed':
        couplings = np.array(find_trace_couplings(grtrh_traces, extr_traces, method='pcc'))
    # sns.set_style("white")
    footprints_cc = np.zeros((len(couplings), len(couplings)))
    traces_cc = np.zeros((len(couplings), len(couplings)))
    for idx1 in range(len(couplings)):
        for idx2 in range(idx1, len(couplings)):
            # print(np.shape(grtrh_footprints))
            img1 = grtrh_footprints[idx1]
            img2 = extr_footprints[int(couplings[idx2][1])]
            tr1 = grtrh_traces[idx1]
            tr2 = extr_traces[int(couplings[idx2][1])]

            if method=='zncc':
                footprints_cc[idx2, idx1] = zero_norm_crosscorr(img1, img2)
                traces_cc[idx2, idx1] = zero_norm_crosscorr(tr1, tr2)
                if not halfmatrix:
                    footprints_cc[idx1, idx2] = zero_norm_crosscorr(img2, img1)
                    traces_cc[idx1, idx2] = zero_norm_crosscorr(tr2, tr1)

            elif method=='mixed':
                footprints_cc[idx2, idx1] = zero_norm_crosscorr(img1, img2)
                traces_cc[idx2, idx1] = pearson_crosscorr(tr1, tr2)
                if not halfmatrix:
                    footprints_cc[idx1, idx2] = zero_norm_crosscorr(img2, img1)
                    traces_cc[idx1, idx2] = pearson_crosscorr(tr2, tr1)

    return traces_cc, footprints_cc
