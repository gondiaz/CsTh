#----------------------
#  Clarice
#     From pmaps produces peak-hits corrected hdf5
#----------------------


import sys
import os
import time

import tables            as tb
import numpy             as np
import pandas            as pd

#import invisible_cities.database.load_db   as db
#import invisible_cities.io.pmaps_io        as pmio
#from   invisible_cities.io.dst_io          import load_dst

#import krcal.dev.corrections               as corrections

#import csth.utils.pmaps_functions           as pmapsf
import csth.utils.cepeak_pmaps_functions as cpkpmap
#import csth.utils.hpeak_hdsts_newfunctions  as hphdst


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-it" , "--input-type"      , type = str ,
                    help = "input data type (e.g. pmaps, hdsts, dfpmaps)",
                    default = 'pmaps'
                    )
parser.add_argument("-r"  , "--run-number"      , type=int  , help="run number")
parser.add_argument("-p"  , "--part-limits"     , type=int  , help="partition limits", nargs=2)
parser.add_argument("-id" , "--input-directory" , type=str  , help="input directory" , default='')
parser.add_argument("-od" , "--output-directory", type=str  , help="output directory", default='')
parser.add_argument("-cf" , "--correction-file" , type=str  , help="corrections file", default='')
parser.add_argument("-q0" , "--q0min"           , type=float, help='q0min', default = cpkpmap.Q0MIN)
parser.add_argument("-full" , "--full"          , type=bool,  help='full output', default = False)
parser.add_argument("-tf" , "--file-tail"       , type=str  , help="tail of file (e.g. Cs, Tlds, Tlpk ...)")

args = parser.parse_args(sys.argv[1:])

input_type       = args.input_type
run_number       = args.run_number
partlim          = args.part_limits
input_directory  = args.input_directory
output_directory = args.output_directory
correction_file  = args.correction_file
file_tail        = args.file_tail
q0min            = args.q0min
full             = args.full

#input_directory = f"$IC_DATA/{run_number}/{}" if input_directory == ''
#output_directory = f"$IC_DATA" if input_directory == ''
#correction_filename = f"$IC_DATA/maps/kr_corrections_run{run_number}.h5" if correction_file = ''


#if input_type == 'pmaps':
ipars   = range(partlim[0], partlim[1] + 1)
spars  = ["{:04}".format(i) for i in range(partlim[0], partlim[1])]
input_files = ["{}/pmaps_{}_{}_{}.h5"  .format(input_directory, spar, run_number, file_tail)
               for spar in spars]
sq0min = str(int(q0min)) + 'q0min'
output_file = "{}/cepks_{}_{}_{}_{}.h5".format(output_directory, run_number, spars[0], spars[-1], sq0min)


#correction_filename = f"$IC_DATA/maps/kr_corrections_run{run_number}.h5"
#correction_filename = os.path.expandvars(correction_filename)
#else: continue
    #JOSE ANGEL SHOULD MODIFY THIS


nfiles = len(input_files)

print(' ')
print(' --- CLARICE  -- ')
print(' run number       : ',run_number)
print(' input_type       : ', input_type)
print(' n input files    : ', nfiles)
print(' first input file : ', input_files[0])
print(' last  input file : ', input_files[-1])
print(' output file      : ', output_file)
print(' correction file  : ', correction_file)
print(' q0min            : ', q0min)
print(' full-ouput       : ', full)
print(' ')

_clarice = cpkpmap.esum



xtime = []
ntotal, naccepted = 0, 0
for iloc, ifile in zip(ipars, input_files):
    xtinit = time.time()

    #counters = 1, 1
    counters, _ = _clarice(ifile, output_file, correction_file, run_number, iloc,
                           q0min = q0min, full = full)

    itot, iacc = counters
    ntotal += itot; naccepted += iacc
    xtend = time.time()
    xtime.append(xtend - xtinit)

f = 100.*naccepted /(1.*ntotal) if ntotal > 0 else 0.
print('total events ', ntotal, ', accepted  ', naccepted, 'fraction (%)' , f)
xtime = np.array(xtime)
print('time per file ', np.mean(xtime), ' s')
if (naccepted <= 1): naccepted = 1
print('time per event', np.sum(xtime)/(1.*naccepted), 's')
