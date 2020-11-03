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
            
            trial_results[key] = lt.tinder_filter(strain_white, lt.colgate(templates[key], spec = np.sqrt(ps)))
            if (np.max(np.abs(trial_results[key]))/np.std(trial_results[key])) > sig_thres:
                print((np.max(np.abs(trial_results[key]))/np.std(trial_results[key])))
                plt.plot(trial_results[key])
                plt.show()
        
        
        ##and save to total results:
        results[data_fname][template_fname] = trial_results
    results[data_fname]["ps"] = ps
##ok so now we ran the matched filter on all the things we have everything we need to finish this up

##

        