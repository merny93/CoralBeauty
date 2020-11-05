import numpy as np
import ligo_tools as lt
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

import sys
sys.stdout = open('output/ligo.txt', 'w')

##first lets get all the the data

data_dir = "data"
key_dict = lt.load_key(os.path.join(data_dir, "BBH_events_v3.json"))



## init results dict
results = {}

sig_thres = 10 #minimum for detection

##lets run the pre-whtining and match filtering on everything
for event_name in key_dict: #loop through each data set
    print("working on event: ", event_name)
    for detector in ["fn_H1", "fn_L1"]: #choose haford or livingston
        print("working on detector: ", detector)
        data_fname = key_dict[event_name][detector]
        data_full_fname = os.path.join(data_dir, data_fname)
        print("reading: " , data_full_fname) 

        ##results formating
        results[event_name + detector] = {}

        strain,dt,utc=lt.read_file(data_full_fname)
        dt = dt/strain.size
        ps, strain_white = lt.colgate(strain)

        template_fname = key_dict[event_name]["fn_template"]
        template_full_fname = os.path.join(data_dir, template_fname)
        print("reading: ", template_full_fname)

        th,tl=lt.read_template(template_full_fname)
        ##for simplicity lets stick the two templates in a dict:
        templates = {"fn_H1" : th, "fn_L1" : tl}

        ##oke so now we have strin and templates.

        

        trial_results = {}



        #call the matched filter 
        template_white = lt.colgate(templates[detector], spec = np.sqrt(ps))
        matched = lt.tinder_filter(strain_white, template_white )
        
        plt.clf()
        plt.plot(matched)
        plt.title(event_name + detector)
        plt.savefig("output/"+event_name + detector+".png")
        ##init results
        trial_results = {}

        ##lets get the the width of the peak to see where it could have actually started 
        max_pos = np.argmax(np.abs(matched)) #this is is some nobody knows units (in array where is max?)

        #im restricting to around the peak just for sanity
        ar_width = np.count_nonzero(np.abs(matched[max_pos - 100: max_pos + 100]) > np.max(np.abs(matched))*0.65)
        trial_results["Positional uncertainty (s)"]  = ar_width *dt
        #i went with 65% confidence interval which is about 1 sigma as its give or take a gaussian
        #so we can tell time differences appart to this resolution. If we want to get a angular resolution it is much more complex.
        #lets say that the wave is coming perpendicular and the two detectors are about a earth diameter appart. 
        # Thus the distnace that light traveels in the uncertainty is directly the sin of the angular uncertainty (over the radius of the earth) 
        # so we can say theta_sigma = sin(uncertainty (s) * c (m/s) / diameter of earth (m)) = unc * c / D_earth
        trial_results["Positional uncertainty (theta)"] = trial_results["Positional uncertainty (s)"] * 3e8 / 12e6

        #however as the signal alligns to the line of the that connects the detectors the uncertainty in angle grows by a bit (not too much but) this is a good order of mangitude tho :)
    


        ##getting the signal to noise is really easy       
        ##just divide max by std but
        ##we want to exclude the peak so it doesnt mess with the std calculation
        noise_loc = list(set(range(matched.size)) - set(np.arange(max_pos - 5*ar_width, max_pos + 5*ar_width)))
        trial_results["signal to noise"] = np.max(np.abs(matched))/np.std(np.take(matched, noise_loc))
        trial_results["delta chi^2"] = trial_results["signal to noise"]**2 ##simply square to get delta chi^2
        
        #this needs to be sqrt(A.T N^{-1} A)
        # But i went through all this effort to whiten the noise!!!
        ##soooo lets make use of it. For us Ninv is very much diagonal and should be give or take 1 if i did my job right
        Ninv = 1
        noise_estimate = np.sqrt(Ninv * np.mean(template_white**2))
        trial_results["analytic noise"] = noise_estimate
        ##we can get a signal to noise estimate once again to compare apples to apples
        trial_results["analytic signal to noise"] = np.max(np.abs(matched))/noise_estimate
        ##now these dont match up excatly. I honestly am not sure why that is but they are very close which is a promising sign
        ##i would tend to trust the other one more cause its a more direct calculation in my mind
       
        ##finally we can get the median frequecy contribution thing which comes from:
        ft_model = np.abs(np.fft.rfft(template_white))**2
        ft_cumsum = np.cumsum(ft_model)
        mid_freq_arg = np.argmin(np.abs(ft_cumsum - ft_cumsum[-1]/2))
        freqs = np.fft.fftfreq(template_white.size, d=dt)
        trial_results["median frequency (Hz)"] = freqs[mid_freq_arg]
    

        
        
        ##and save to total results:
        results[event_name + detector] = trial_results
    total_chi = results[event_name + "fn_H1"]["delta chi^2"] + results[event_name + "fn_L1"]["delta chi^2"]
    results[event_name + detector]["total chi^2 for both detectors: "] = total_chi
    #results[event_name + detector]["ps"] = ps
for key in results:
    print("results for data run: ", key)
    headers = results[key].keys()
    values = [[results[key][x]] for x in results[key]]
    to_print = dict(zip(headers, values))
    print(tabulate(to_print, headers="keys"))
##ok so now we ran the matched filter on all the things we have everything we need to finish this up with printing the results 


