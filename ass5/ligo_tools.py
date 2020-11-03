import numpy as np
from h5py import File


def my_window(N, option= None):
    """
    Return a window of length N 
    
    Defaults to hanning but if called with options ["Tukey", frac]
    Will give a wider window such that it only windows the outer frac 
    of the data

    """
    if option is None:
        return np.hanning(N)
    if option[0] == "Tukey": ##Tukey window this like the cos window but has flat 1 on top so is good shit
        window = np.hanning(int(N * option[1]))
        window_full = np.ones(N)
        window_full[:window.size//2] = window[: window.size //2]
        window_full[-window.size//2:] = window[-window.size //2 :]
        return window_full 



def colgate(data, window = my_window, window_params=["Tukey", 0.02], spec= None, plot_model =False):
    '''
    Pre-whiten data to get it ready for match filtering

    If called with no spectrum it will assume that you want to generate the noise spectrum and do so 
    It will use a window function which defualts to a hanning but can be changed to wider functions 


    '''
    if spec is None:
        ##get the ft to look at noise
        ft_noise = np.abs(((np.fft.rfft(data * window(data.size, window_params)))))**2
    
        if plot_model:
            plt.clf()
            plt.loglog(ft_noise, label= "before")

        ##what we want to do is draw a line along the peaks to make sure its not under represented 
        ## to do this i will make a rolling window of 3 elements and if the middle is smaller than both of its neigbors 
        ## i will replace it with the average of the two neigbhors.
        ## And then do this a bunch of times. 
        ##the construction here will mean that we never loose the peaks (cause we always take maximums) 
        # but the valleys will get pulled up to the tops around them
        # if we run this long enough it will slowly become a straight line between the two biggest peaks with lines from the edges going there
        #but it will never get there since we dont run it for long enough
        
        niter = 500 #number of iterations (this is a guess but works ok)
        for _ in range(niter):
            #I started with a python loop implementation and it took forever so sorry for the hard code 
            # I want to make a rolling window array. So 3 wide such that the first row is 1st ,2nd ,3rd element 
            #second row is 2nd, 3rd, 4th element and so on so forth 
            #numpy does not have a funciton to do this so lets tell it to read the array differently with strides!

            stride = (ft_noise.strides[0], ft_noise.strides[0])
            #this line tells python that a move to the right or a move to down is the same equivalent 
            #since the elemnt to the right and the element bellow are both indexed 1 away in the original array

            ft_rolling = np.lib.stride_tricks.as_strided(ft_noise, shape = (ft_noise.size - 2, 3), strides=stride, writeable=False)
            ##that line generated the array representation (not actually writable)

            ft_argmin = np.argmin(ft_rolling, axis=-1)#check which is the minimum 

            ft_reset = np.where(ft_argmin == 1, (ft_rolling[:,0] + ft_rolling[:,2])/2, ft_rolling[:,1])
            #another hard line. This one checks if the min happened in the middle and if that is the case fills an array
            #of size ft_noise.size -2 with the average of the points around or just copies the value if it wasnt the min

            ft_noise[1:-1] = ft_reset #finally adding it to the array for the next itteration
            
            
            continue
            ##here is the original (almost) identical code
            print("this is demonstration shouldnt run")
            for i in range(ft_noise.size -2):
                if ft_noise[i] > ft_noise[i+1] and ft_noise[i+2] > ft_noise[i+1]:
                    ft_noise[i+1] = (ft_noise[i] + ft_noise[i+2])/2


        PS = ft_noise
        if plot_model:
            plt.loglog(ft_noise, label = "after")
            plt.legend()
            plt.savefig("output/model_dome.png")
        
        return PS, colgate(data, window_params=window_params, spec = np.sqrt(PS))
    
    ft_white = (np.fft.rfft(data * window(data.size, window_params))) / spec
    data_white = np.fft.irfft(ft_white)
    return data_white

def tinder_filter(y, model):
    '''
    a matching service for pre-whitened signal

    '''
    yft=np.fft.rfft(y)
    modft=np.fft.rfft(model)
    mycorr=np.fft.irfft(yft*np.conj(modft))
    return mycorr


##jon code for reading here
def read_template(filename):
    dataFile=File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



if __name__=="__main__":
    import matplotlib.pyplot as plt

    fname= "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
    print('reading file ',fname)
    strain,dt,utc=read_file(fname)

    template_name= "data/GW150914_4_template.hdf5"
    th,tl=read_template(template_name)



    ##the 0.02 width comes from trial and error. Thats the biggest value such that we stop seing 
    ## the signal attenuation (if we go much smaller its gonna start spectral leakage again)
    #lets plot with a bigger frac to see what is going on
    plt.clf()
    plt.plot(my_window(100, ["Tukey", 0.25] ))
    plt.savefig("output/window_shape.png")

    ps, strain_white = colgate(strain, window_params=["Tukey", 0.02], plot_model= True )

    th_white = colgate(th , window_params=["Tukey", 0.02], spec=np.sqrt(ps))

    tl_white = colgate(tl, window_params=["Tukey", 0.02], spec=np.sqrt(ps))

    plt.clf()
    plt.plot(np.cumsum(np.abs(np.fft.rfft(tl_white))))
    plt.show()

    th_fit = tinder_filter(strain_white, th_white)

    tl_fit = tinder_filter(strain_white, tl_white)

    print("detection with signal to noise: ", np.max(np.abs(tl_fit))/np.std(tl_fit))
    plt.clf()
    plt.plot(tl_fit)
    plt.savefig("output/sample_detection.png")

