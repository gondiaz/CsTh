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

import invisible_cities.database.load_db   as db
import invisible_cities.io.pmaps_io        as pmio
from   invisible_cities.io.dst_io          import load_dst

import krcal.dev.corrections               as corrections


import csth.utils.pmaps_functions           as pmapsf
import csth.utils.hpeak_pmaps_newfunctions  as hppmap
import csth.utils.hpeak_hdsts_newfunctions  as hphdst


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

    input_type  = str(args[1])
    run_number  = int(args[2])
    if (input_type == 'pmaps'):
        mode  = ''
        file_trail = str(args[3])
        iini       = int(args[4])
        iend       = int(args[5])
        partitions       = ["{:04}".format(i) for i in range(iini, iend)]
        input_filenames  = [f"$IC_DATA/{run_number}/pmaps/trigger2/pmaps_{par}_{run_number}_{file_trail}.h5" for par in partitions]
        #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
        input_files      = [os.path.expandvars(fi) for fi in input_filenames]
        output_filename  = f"$IC_DATA/{run_number}/pmaps/edf_{run_number}_{iini}_{iend-1}.h5"
        output_file      = os.path.expandvars(output_filename)
        return (run_number, input_files, output_file, input_type)

    if (input_type == 'hdsts'):
        mode  = ''
        file_trail = str(args[3])
        iini       = int(args[4])
        iend       = int(args[5])
        partitions       = ["{:04}".format(i) for i in range(iini, iend)]
        input_filenames  = [f"$IC_DATA/{run_number}/hdsts/trigger2/hdst_{par}_{run_number}_{file_trail}.h5" for par in partitions]
        #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
        input_files      = [os.path.expandvars(fi) for fi in input_filenames]
        output_filename  = f"$IC_DATA/{run_number}/hdsts/edf_{run_number}_{iini}_{iend-1}.h5"
        output_file      = os.path.expandvars(output_filename)
        return (run_number, input_files, output_file, input_type)

    #if (city == 'pmaps_gd'):
    #    mode    = 'gc'
    #    input_file  = str(args[3])
    #    output_file = str(args[4])
    #    input_files = [os.path.expandvars(input_file),]
    #    output_file =  os.path.expandvars(output_file)
    #    return (run_number, input_files, output_filename, read_fun)

    if (input_type == 'pmaps_gd'):
        read_fun    = 'gd'
        file_head  = str(args[3])
        iini       = int(args[4])
        iend       = int(args[5])
        partitions          = [str(i) for i in range(iini, iend+1)]
        input_filenames     = [f"$IC_DATA/{run_number}/pmaps/trigger2/{file_head}_{run_number}_{par}.h5" for par in partitions]
        #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
        input_files         = [os.path.expandvars(fi) for fi in input_filenames]
        output_filename     = f"$IC_DATA/{run_number}/pmaps/edf_{file_head}_{run_number}_{iini}_{iend}.h5"
        output_file =  os.path.expandvars(output_filename)
        return (run_number, input_files, output_file, input_type)

"""
def get_files(i0, i1):
    partitions          = ["{:04}".format(i) for i in range(i0, i1+1)]
    input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_head}_{par}_{file_trail}.h5" for par in partitions]
    #input_filenames     = [f"$IC_DATA/{run_number}/pmaps/{file_par}.h5" for par in partitions]
    input_files         = [os.path.expandvars(fi) for fi in input_filenames]
    output_filename     = f"pkhits_{run_number}_{i0}_{i1-1}.h5"
    return input_files, output_filename
"""

def _partition(filename):
    words = filename.split('_')
    partition = int(words[1])
    return partition


def _clarice_pmaps(file, xpos, ypos, calibrate, output_filename):
    try:
        #pmaps = pmapsf.get_pmaps(file, '')
        pmaps   = pmapsf.get_pmaps(file, '')
        runinfo = load_dst(file, 'Run', 'events')
    except:
        return 0., 0.
    partition = _partition(file)
    edf = hppmap.events_summary(pmaps, runinfo, partition, xpos, ypos, calibrate)
    edf .to_hdf(output_filename, key = 'edf'   , append = True)
    itot, iacc = len(set(pmaps.s1.event)), len(set(edf.event))
    return itot, iacc

def _clarice_pmaps_gd(file, xpos, ypos, calibrate, output_filename):
    try:
        pmaps = pmapsf.get_pmaps(file, mode, 'gd')
    except:
        return 0., 0.
    partition = _partition(file)
    edf = hppmap.events_summary(pmaps, partition, xpos, ypos, calibrate)
    edf .to_hdf(output_filename, key = 'edf'   , append = True)
    itot, iacc = len(set(pmaps.s1.event)), len(set(edf.event))
    return itot, iacc

def _clarice_hdsts(file, calibrate, output_filename):
    hits = load_dst(file,'RECO','Events')
    try:
        nevents = len(hits.event)
        #print('nevents ', nevents)
        if (nevents <= 0): raise
    except:
        print('not loaded file ', file)
        return 0., 0.
    partition = _partition(file)
    edf = hphdst.events_summary(hits, partition, calibrate)
    edf .to_hdf(output_filename, key = 'edf'   , append = True)
    itot, iacc = len(set(hits.event)), len(set(edf.event))
    return itot, iacc


def clarice(input_type, run_number, input_filenames, output_filename):

    # initalize
    correction_filename = f"$IC_DATA/maps/kr_corrections_run{run_number}.h5"
    correction_filename = os.path.expandvars(correction_filename)
    calibrate = corrections.Calibration(correction_filename, 'scale')

    datasipm = db.DataSiPM(run_number)
    sipms_xs, sipms_ys = datasipm.X.values, datasipm.Y.values

    ntotal, naccepted  = 0, 0

    xtime = np.zeros(len(input_filenames))
    for i, file in enumerate(input_filenames):

        xtinit = time.time()
        print('processing ', file)
        if (input_type == 'pmaps'):
            itot, iacc = _clarice_pmaps(file, sipms_xs, sipms_ys, calibrate, output_filename)
            ntotal += itot; naccepted += iacc
        elif (input_type == 'pmaps'):
            itot, iacc = _clarice_pmaps(file, sipms_xs, sipms_ys, calibrate, output_filename)
            ntotal += itot; naccepted += iacc
        elif (input_type == 'hdsts'):
            itot, iacc = _clarice_hdsts(file, calibrate, output_filename)
            ntotal += itot; naccepted += iacc
        xtend = time.time()
        xtime[i] = xtend - xtinit


    f = 100.*naccepted /(1.*ntotal) if ntotal > 0 else 0.
    print('total events ', ntotal, ', accepted  ', naccepted, 'fraction (%)' , f)
    print('time per file ', np.mean(xtime), ' s')
    if (naccepted <= 1): naccepted = 1
    print('time per event', np.sum(xtime)/(1.*naccepted), 's')
    #f = 100.*ngoodpeaks/(1.*npeaks) if npeaks > 0 else 0.
    #print('total peaks  ', npeaks, ', good hits ', ngoodpeaks, 'fraction (%)', f)

#-----------

run_number, input_files, output_file, input_type = arguments(sys.argv)

print(' ')
print(' CLARICE : ', run_number)
print(' type    : ', input_type)
print(' inputs  : ', len(input_files), 'file: ', input_files[0])
print(' output  : ', output_file)
print(' ')

clarice(input_type, run_number, input_files, output_file)
