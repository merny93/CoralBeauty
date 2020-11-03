import numpy as np
import ligo_tools as lt
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

##first lets get all the the data

data_dir = "data"

fnames = os.listdir(data_dir)
template_fnames = [fname for fname in fnames if fname.split("_")[-1] == "template.hdf5"]
data_fnames = list(set(fnames) - set(template_fnames)) #set difference in python wot wot

## init results dict
results = {}

sig_thres = 10 #minimum for detection

##lets run the pre-whtining and match filtering on everything
for data_fname in data_fnames: #loop through each data set
   
    data_full_fname = os.path.join(data_dir, data_fname)
    print("reading: " , data_full_fname) 

    ##results formating
    results[data_fname] = {}

    strain,dt,utc=lt.read_file(data_full_fname)
    ps, strain_white = lt.colgate(strain)

    for template_fname in template_fnames:
        template_full_fname = os.path.join(data_dir, template_fname)
        print("reading: ", template_full_fname)

        th,tl=lt.read_template(template_full_fname)
        ##for simplicity lets stick the two templates in a dict:
        templates = {"th" : th, "tl" : tl}

        ##oke so now we have strin and templates.

        

        trial_results = {}

        for key in templates: #never be afraid to add a loop that saves like 1 line and make the code less readable 
            #this simply loops through the th and tl templates

            #call the matched filter 
            template_white = lt.colgate(templates[key], spec = np.sqrt(ps))
            matched = lt.tinder_filter(strain_white, template_white )

            if (np.max(np.abs(matched))/np.std(matched)) > sig_thres: #check for event
                ## we have an event with statistical significance 

                ##init results
                trial_results[key] = {}

                ##lets get the the width of the peak to see where it could have actually started 
                trial_results[key]["position"] = np.argmax(np.abs(matched))
                trial_results[key]["width"] = np.count_nonzero(np.abs(matched) > np.max(np.abs(matched))*0.65)
                #i went with 65% confidence interval which is about 1 sigma as its give or take a gaussian

                ##getting the signal to noise is really easy       
                ##just divide max by std but
                ##we want to exclude the peak so it doesnt mess with the std calculation
                noise_loc = list(set(range(matched.size)) - set(np.arange(trial_results[key]["position"] - trial_results[key]["width"], trial_results[key]["position"] + trial_results[key]["width"])))
                trial_results[key]["signal to noise"] = np.max(np.abs(matched))/np.std(np.take(matched, noise_loc))
                
                #now estimated noise is given here which i dont know how to do so problem for tmr :)
                #this needs to be sqrt(A.T N^{-1} A)
                # but we went through a bunch of effort to pre whiten so we can take to be diagonal and to be std of strain_white
                Ninv = 1/ np.std(strain_white)

                noise_estimate = np.sum(np.abs(template_white))*np.sqrt(Ninv)
                noise_estimate = np.max(np.abs(matched))/np.std(template_white)
                trial_results[key]["analytic noise"] = noise_estimate

                ##finally we can get the frequecy thing which comes from:
                ft_model = np.abs(np.fft.rfft(template_white))
                ft_cumsum = np.cumsum(ft_model)
                trial_results[key]["median frequency"] = np.argmin(np.abs(ft_cumsum - ft_cumsum[-1]/2))
            
                for quant in trial_results[key]:
                    print(quant, trial_results[key][quant])
                

        
        
        ##and save to total results:
        results[data_fname][template_fname] = trial_results
    results[data_fname]["ps"] = ps
##ok so now we ran the matched filter on all the things we have everything we need to finish this up with printing the results 


