#!/usr/bin/env python3

# This is a simple Python script - want it to be runnable without any nonstandard packages
import os
import sys
import subprocess
from datetime import datetime, timedelta

def run_multi(dirname, sw_input_vec, lw_input_vec, cld_input_vec, sw_bin, lw_bin,
              sw_output_short_vec,lw_output_short_vec,verbose=False):

    bash_cmd_vec = ['#!/bin/sh']
    sw_out_vec = []
    lw_out_vec = []
    for sw_input, lw_input, cld_input, sw_out, lw_out in zip(sw_input_vec,lw_input_vec,cld_input_vec,
                                                             sw_output_short_vec,lw_output_short_vec):
        # Skip cases where no shortwave input is found - this happens if the SZA is too high
        skip_sw = not os.path.isfile(os.path.join(dirname,sw_input))
        
        # Commands to prepare input files, run RRTM, and then move the output files
        lw_prep = 'unlink INPUT_RRTM; unlink IN_CLD_RRTM; ln -s {:s} INPUT_RRTM; ln -s {:s} IN_CLD_RRTM'.format(lw_input, cld_input)
        lw_bash = '( echo {:s}; echo {:s}; echo "_lw_apcemm"; echo "./"; echo 2 ) | {:s}'.format(lw_input,cld_input,lw_bin)
        lw_move = 'mv OUTPUT-RRTM_lw_apcemm ' + lw_out
        LW_output_path = os.path.join(dirname,lw_out)
        
        cmd_list_lw = [lw_prep, lw_bash, lw_move]
        
        # Commands to prep for the run
        if skip_sw:
            sw_prep = 'unlink INPUT_RRTM; unlink IN_CLD_RRTM; ln -s {:s} INPUT_RRTM; ln -s {:s} IN_CLD_RRTM'.format(sw_input, cld_input)
            sw_bash = '( echo {:s}; echo {:s}; echo "_sw_apcemm"; echo "./"; echo 2 ) | {:s}'.format(sw_input,cld_input,sw_bin)
            sw_move = 'mv OUTPUT_RRTM ' + sw_out
            SW_output_path = os.path.join(dirname,sw_out)
            cmd_list_sw = [sw_prep, sw_bash, sw_move]
        else:
            SW_output_path = None
            cmd_list_sw = []
        

        sw_out_vec.append(SW_output_path)
        lw_out_vec.append(LW_output_path)

        for cmd in cmd_list_sw + cmd_list_lw:
            bash_cmd_vec.append(cmd)

    bash_cmd = '\n'.join(bash_cmd_vec)
    
    # Go to the directory containing the data. Easiest way to handle the fact that the SW code reads from
    # INPUT_RRTM regardless of what file is specified as input
    start_dir = os.getcwd()
    try:
        os.chdir(dirname)
        
        # Write the command list to file
        f_batch = 'batch_rrtm.sh'
        if os.path.isfile(f_batch):
            os.remove(f_batch)
        with open(f_batch,'w') as f:
            for l in bash_cmd_vec:
                f.write(f"{l}\n")
        
        # Owner rwx, everyone else r only
        # Leading 0o means octal
        os.chmod(f_batch,0o744)
        #with open(f_batch,'r') as f:
        #    print(f.read())
        #print(os.getcwd())
        
        process = subprocess.Popen("./{:s}".format(f_batch), stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if verbose:
            print(output.decode())
            print(error.decode())
        
    finally:
        os.chdir(start_dir)
        
    raise ValueError
    return lw_out_vec, sw_out_vec

def run_single(dirname, sw_input, lw_input, cld_input, sw_bin, lw_bin,
               sw_output_short='sw_output',lw_output_short='lw_output',verbose=False):
    # Build the bash commands for each
    sw_bash = '( echo {:s}; echo {:s}; echo "_sw_apcemm"; echo "./"; echo 2 ) | {:s}'.format(sw_input,cld_input,sw_bin)
    lw_bash = '( echo {:s}; echo {:s}; echo "_lw_apcemm"; echo "./"; echo 2 ) | {:s}'.format(lw_input,cld_input,lw_bin)
    
    LW_output_path = os.path.join(dirname,lw_output_short)
    SW_output_path = os.path.join(dirname,sw_output_short)
    
    # Go to the directory containing the data. Easiest way to handle the fact that the SW code reads from
    # INPUT_RRTM regardless of what file is specified as input
    start_dir = os.getcwd()
    try:
        os.chdir(dirname)
        if verbose:
            print(lw_bash)
        process = subprocess.Popen(lw_bash, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if verbose:
            print(output.decode())
            print(error.decode())
        os.rename('OUTPUT-RRTM_lw_apcemm',lw_output_short)

        if os.path.isfile('INPUT_RRTM'):
            os.remove('INPUT_RRTM')
        if os.path.isfile('IN_CLD_RRTM'):
            os.remove('IN_CLD_RRTM')

        if os.path.isfile(sw_input):
            os.symlink(sw_input,'INPUT_RRTM')
            os.symlink(cld_input,'IN_CLD_RRTM')
            if verbose:
                print(sw_bash)
            # Make sure the symlink is deleted even if the process fails
            try:
                process = subprocess.Popen(sw_bash, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, shell=True)
                output, error = process.communicate()
                if verbose:
                    print(output.decode())
                    print(error.decode())
            finally:
                os.unlink('INPUT_RRTM')
                os.unlink('IN_CLD_RRTM')
            os.rename('OUTPUT_RRTM',sw_output_short)

        for f in ['tape6','TAPE7']:
            if os.path.isfile(f):
                os.remove(f)
    finally:
        os.chdir(start_dir)
    return LW_output_path, SW_output_path

def run_directory(dirname, sw_bin=None, lw_bin=None, verbose=False, use_single=False):

    if sw_bin is None:
        sw_bin = './rrtmg_sw'

    if lw_bin is None:
        lw_bin = './rrtmg_lw'

    # Figure out how many cases we have
    # Use longwave because shortwave files are not generated when SZA is too high
    all_lw_input = [x for x in os.listdir(dirname) if x.startswith('lw_input_t') and not os.path.isdir(x)
                                                      and x.endswith('_clr')]
    
    if not use_single:
        sw_out_vec = []
        lw_out_vec = []
        sw_in_vec = []
        lw_in_vec = []
        cld_in_vec = []
        
    for lw_clr in all_lw_input:
        f_split = lw_clr.split('_')
        tstamp  = f_split[2]
        column  = f_split[3]
        # Run two cases: with and without contrail
        for cld_case in ['cld','clr']:
            sw_out = 'sw_output_{:s}_{:s}_{:s}'.format(tstamp,column,cld_case)
            lw_out = 'lw_output_{:s}_{:s}_{:s}'.format(tstamp,column,cld_case)
            sw_in  = 'sw_input_{:s}_{:s}_{:s}'.format(tstamp,column,cld_case)
            lw_in  = 'lw_input_{:s}_{:s}_{:s}'.format(tstamp,column,cld_case)
            cld_in = 'cld_input_{:s}_{:s}_{:s}'.format(tstamp,column,cld_case)
            if use_single:
                run_single(dirname,sw_in,lw_in,cld_in,
                           sw_bin,lw_bin,sw_out,lw_out,
                                 verbose=verbose)
            else:
                sw_out_vec.append(sw_out)
                lw_out_vec.append(lw_out)
                sw_in_vec.append(sw_in)
                lw_in_vec.append(lw_in)
                cld_in_vec.append(cld_in)
        
    if not use_single:
        run_multi(dirname, sw_in_vec, lw_in_vec, cld_in_vec, sw_bin, lw_bin,
              sw_out_vec,lw_out_vec,verbose=verbose)
        
    return len(all_lw_input)

if __name__ == '__main__':
    from time import time
    verbose = False
    for dirname in sys.argv[1:]:
        print('Processing ' + dirname)
        t_start = time()
        n_cases = run_directory(dirname,verbose=verbose)
        t_end   = time()
        t_elapsed = t_end - t_start
        ms_per_case = t_elapsed / n_cases
        print(' --> Processed {:d} cases in {:.1f} seconds ({:.1f} s/case)'.format(n_cases,t_elapsed,ms_per_case))
