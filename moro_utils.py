import sys
import matplotlib.pyplot as plt
import numpy as np
import skvideo

def printProgressBar (iteration, total, prefix ='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
#     print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
#     outstr = '\r%s |%s| %s%% %s : %i/%i' % (prefix, bar, percent, suffix, iteration, total)
    outstr = '\r%s |%s| : %i/%i' % (prefix, bar, iteration, total)
#     '{} {}'.format(1, 2)
    sys.stdout.write(outstr)
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

def saveVideo(videoMtrx, outfile):
    # make videos
    out_vstream = skvideo.io.FFmpegWriter(outfile, outputdict={
          '-vcodec': 'libx264',
          '-pix_fmt': 'yuv420p',
          '-r': '9',
    })
    for frameno in range(videoMtrx.shape[0]):
    #     if not make a copy, after the writeFrame call the variable will be casted to a uint8
    #     without scaling, so will become a matrix with just 255 in all the entries
        out_vstream.writeFrame(copy.deepcopy(videoMtrx[frameno,:,:]))
    out_vstream.close()

def show_video(video, figsize=(10,10), autoscale=False):
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig, ax1 = plt.subplots(1, figsize=figsize,  constrained_layout=True)
    im = ax1.imshow(np.max(video, axis=0), cmap='gray')
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0., hspace=0., wspace=0.)
    ax1.set_axis_off()
    idx = 0
    tot_frames = np.shape(video)[0]

    def updatefig(idx):
        im.set_array(video[idx])
        if autoscale:
            im.autoscale()
        # ax1.set_title('Frame ' + str(idx))
        return fig,
                
    # steps = np.arange(tot_frames)

    ani = animation.FuncAnimation(fig, updatefig, frames=tot_frames, interval=250, blit=True)
    return HTML(ani.to_html5_video())
    # plt.show()
    # pass

# def show_video(video):
#     import matplotlib.animation as animation
#     from IPython.display import HTML

#     fig, ax1 = plt.subplots(1, figsize=(10,10))
#     im = ax1.imshow(np.max(video, axis=0), cmap='gray')
#     ax1.set_axis_off()
#     idx = 0
#     tot_frames = np.shape(video)[0]

#     def updatefig(idx):
#         im.set_array(video[idx])
#         # ax1.set_title('Frame ' + str(idx))
#         return fig,
                
#     # steps = np.arange(tot_frames)

#     ani = animation.FuncAnimation(fig, updatefig, frames=tot_frames, interval=250, blit=True)
#     return HTML(ani.to_html5_video())
#     # plt.show()
#     # pass

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def shiftimg(img, shift):
    tmp1 = np.zeros_like(img)
    if   (shift[0]>0):
        tmp1[shift[0]:,:] = img[0:-shift[0],:]
    elif (shift[0]<0):
        tmp1[0:shift[0],:] = img[-shift[0]:,:]
    else:
        tmp1 = img
    tmp2 = np.zeros_like(img) 
    if (shift[1]>0):
        tmp2[:,shift[1]:] = tmp1[:, 0:-shift[1]]
    elif (shift[1]<0):
        tmp2[:,0:shift[1]] = tmp1[0:,-shift[1]:]
    return tmp2