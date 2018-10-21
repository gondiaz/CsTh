# CsTh
Code for the Cs/Th NEXT analysis with IC

This respository contains code to correct pmaps and hdsts. It produces a DataFrame with a summary of the event information, including the corrected energy.

The main script is 'clarice.py' in the cities directory. 
It takes as arguments the type of data (pmaps, hdsts), the run number, the trail of the filenames, and the initial and final number of the partion of the files.

for example:
> python clarice.py pmaps 6348 trigger2_v0.9.9_20180921_krbg1300 0 10
it runs on pmaps for the run 6248 for files 0 to 10

The module expect that the data filenames have the name pmaps_{partition}_{run_number}_{filename_trail}.h5 

The output is a h5 file with a DataFrame named 'edf' that contains the summary information of the event, including the corrected energy