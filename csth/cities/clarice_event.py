#----------------------
#  Clarice
#     From pmaps produces peak-hits corrected hdf5
#----------------------


import sys
import os
import time

import tables            as tb
import numpy             as np
import collections       as collections
import pandas            as pd

import csth.utils.hpeak_functions      as hpfun

#file_head  = str(sys.argv[2])
#par_ini, par_end = -1, -1
#if (len(sys.argv) > 3):
#    par_ini    = int(sys.argv[3])
#    par_end    = int(sys.argv[4])
#if (len(sys.argv) > 6 ):
#    file_trail ='_'+str(sys.argv[5])

#output_filename     = f"pkhits_{file_head}.h5"
# tag = file_trail
#run_number = 6348
#tag                 = "trigger2_v0.9.9_20180921_krbg1300"
#partitions          = ["{:04}".format(i) for i in range(5)]
#input_filenames     = [f"$IC_DATA/{run_number}/pmaps/pmaps_{par}_{run_number}_{tag}.h5" for par in partitions]
#input_files         = [os.path.expandvars(fi) for fi in input_filenames]
#correction_filename = f"$IC_DATA/maps/kr_corrections_run{run_number}.h5"
#output_filename     = f"corhits_{run_number}.h5"

def arguments(args):

    city        = str(args[1])
    run_number  = int(args[2])
    if (city == 'pmaps'):
        read_fun  = hpfun.get_pmaps
        file_head  = str(args[3])
        file_trail = str(args[4])
        iini       = int(args[5])
        iend       = int(args[6])
        partitions          = ["{:04}".format(i) for i in range(iini, iend)]
        input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_head}_{par}_{file_trail}.h5" for par in partitions]
        #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
        input_files         = [os.path.expandvars(fi) for fi in input_filenames]
        output_filename     = f"pkhits_{run_number}_{i0}_{i1-1}.h5"
        return (run_number, input_files, output_file, read_fun)


    if (city == 'gmaps'):
        read_fun    = hpfun.get_pmaps_gd
        input_file  = str(args[3])
        output_file = str(args[4])
        input_files = [os.path.expandvars(input_file),]
        output_file =  os.path.expandvars(output_file)
        return (run_number, input_files, output_filename, read_fun)

    if (city == 'gmaps2'):
        read_fun    = hpfun.get_pmaps_gd
        file_head  = str(args[3])
        iini       = int(args[4])
        iend       = int(args[5])
        partitions          = [str(i) for i in range(iini, iend+1)]
        input_filenames     = [f"$IC_DATA/{run_number}/pmaps/trigger2/{file_head}_{run_number}_{par}.h5" for par in partitions]
        #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
        input_files         = [os.path.expandvars(fi) for fi in input_filenames]
        output_filename     = f"$IC_DATA/{run_number}/pmaps/trigger2/pkhit_{file_head}_{run_number}_{iini}_{iend}.h5"
        output_file =  os.path.expandvars(output_filename)
        return (run_number, input_files, output_file, read_fun)


def get_files(i0, i1):
    partitions          = ["{:04}".format(i) for i in range(i0, i1+1)]
    input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_head}_{par}_{file_trail}.h5" for par in partitions]
    #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
    input_files         = [os.path.expandvars(fi) for fi in input_filenames]
    output_filename     = f"pkhits_{run_number}_{i0}_{i1-1}.h5"
    return input_files, output_filename

def clarice(run_number, input_filenames, output_filename, read_fun = hpfun.get_pmaps_gd):

    hits_from_peak = hpfun.functor_hits_from_peak(run_number)

    ntotal, naccepted  = 0, 0
    npeaks, ngoodpeaks = 0, 0
    for file in input_filenames:
        print('processing ', file)
        try:
            #pmaps = hpfun.get_pmaps_gd(file)
            pmaps = read_fun(file)
        #print(pmaps)
        except:
            print('not valid file ', file)
            continue

        evts  = hpfun.pmaps_event_list(pmaps)
        #print(evts)
        ntotal += len(evts)

        for ievt, evt in enumerate(evts):
            pm = hpfun.pmaps_get_event(pmaps, evt)
            n1, n2 = hpfun.pmap_npeaks(pm)
            #print(n1, n2)
            if (n1 != 1 or n2 != 1):
                continue
            naccepted += 1
            for ipeak in range(n2):
                npeaks += 1
                if (npeaks%100 == 0):
                    print('event ', evt, ' s2 ', ipeak)
                ok, iesum, iezs, ihits = hits_from_peak(pm.s1, pm.s2, pm.s2i, s2_peak = ipeak)
                if (ok):
                    ngoodpeaks += 1
                    iesum.to_hdf(output_filename, key = 'pkevent' , append = True)
                    iezs .to_hdf(output_filename, key = 'pkslices', append = True)
                    #ihits.to_hdf(output_filename, key = 'pkhits'  , append = True)
    f = 100.*naccepted /(1.*ntotal) if ntotal > 0 else 0.
    print('total events ', ntotal, ', accepted  ', naccepted, 'fraction (%)' , f)
    f = 100.*ngoodpeaks/(1.*npeaks) if npeaks > 0 else 0.
    print('total peaks  ', npeaks, ', good hits ', ngoodpeaks, 'fraction (%)', f)


#-----------

run_number, input_files, output_file, read_fun = arguments(sys.argv)

print(' ')
print(' CLARICE : ', run_number)
print(' inputs  : ', len(input_files), 'file: ', input_files[0])
print(' output  : ', output_file)
print(' ')

clarice(run_number, input_files, output_file, read_fun)
