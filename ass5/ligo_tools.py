import numpy as np
from h5py import File

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


def my_window(N, option= None):
    if option is None:
        return np.hanning(N)
    if option[0] == "Tukey": ##Tukey window this like the cos window but has flat 1 on top so is good shit
        window = np.hanning(int(N * option[1]))
        window_full = np.ones(N)
        window_full[:window.size//2] = window[: window.size //2]
        window_full[-window.size//2:] = window[-window.size //2 :]
        return window_full 

def colgate(data, window = my_window, window_params=None, spec= None):
    '''
    Pre-whiten data to get it ready for match filtering

    '''
    if spec is None:
        ft_noise = (((np.fft.rfft(data * window(data.size, window_params)))))**2
        ft_cumsum = np.cumsum(ft_noise)
        ft_smooth = (ft_cumsum - np.roll(ft_cumsum, 5))/5
        PS = np.maximum(ft_smooth, ft_noise)
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



if __name__=="__main__":
    import matplotlib.pyplot as plt

    fname= "LOSC_Event_tutorial/H-H1_LOSC_4_V2-1126259446-32.hdf5"
    print('reading file ',fname)
    strain,dt,utc=read_file(fname)

    template_name= "LOSC_Event_tutorial/GW150914_4_template.hdf5"
    th,tl=read_template(template_name)



    ps, strain_white = colgate(strain, window_params=["Tukey", 0.2] )

    th_white = colgate(th, spec=np.sqrt(ps))

    tl_white = colgate(tl, spec=np.sqrt(ps))


    th_fit = tinder_filter(strain_white, th_white)

    tl_fit = tinder_filter(strain_white, tl_white)

    plt.plot(th_fit)
    plt.show()
