# Chainlink
#
# A script to automate concatenative synthesis
#
# Raymond Viviano
# January 31st, 2019
# rayvivianomusic@gmail.com

# Dependencies
from __future__ import print_function
import os, sys, getopt, traceback
import wave, sndhdr
import numpy as np 
import multiprocessing as mp
from os.path import isdir, isfile, abspath, join


# Function Definitions
def process_options():
    """
        Process command line arguments for file inputs, outputs, verbosity,
        chunk length, multiprocessing, and number of cores to use. Also provide 
        an option for help. Return input dirs, ouput dir, and verbosity level. 
        Print usage and exit if help requested.
    """

    # Define usage
    usage = """
    Usage: python chainlink.py --input1 <arg> --input2 <arg> --output <arg> 
                               --chunk_size <arg> [-v] [-m] [-c cores] [-h]

    Mandatory Options:

    --input1      Directory containing wav files to recreate with concatenative 
                  synthesis. Can contain other files, but this script will only
                  process the wavs within.

    --input2      Directory conatining the "chain links," or a bunch of wavs that
                  the script will use to recreate the wavs in 'input1'

    --output      Directory where you want the script to save output

    --chunk_size  Number between 10 and 1000. The chunk size in milleseconds, 
                  where a chunk is the segment of a sample from input1 that gets
                  replaced by a segment of the same size from a sample within 
                  the input2 directory

    Optional ARguments:

    -v            Turn verbosity on - increases text output the script generates

    -m            Turn multiprocessing on - leverages multicore systems

    -c            Number of cores to use, defaults to 2 if multiprocessing is 
                  specified but the user doesn't pass an argument to this option

    -h            Print this usage message and exit
    """

    # Set verbosity to false
    is_verbose = False

    # Set multiprocessing to false
    is_mp = False

    # Set number of cores to use for multiprocessing to 2 as a default
    cores = 2

    # Get commandline options and arguments
    options, _ = getopt.getopt(sys.argv[1:], ["input1=", "input2=", "output=",
                                              "chunk_size=", "h", "m", "c", "v"])

    for opt, arg in options:
        if opt == "--input1":
            input_dir1 = arg
        if opt == "--input2": 
            input_dir2 = arg 
        if opt == "--output":
            output_dir = arg 
        if opt == "--chunk_size":
            chunk_size = arg
        if opt == "-v":
            is_verbose = True
        if opt == "-m":
            is_mp = True
        if opt == "-c":
            cores = arg
        if opt == "-h":
            print(usage)
            sys.exit(0)

    # Verify usability of passed arguments
    check_options(input_dir1, input_dir2, output_dir, chunk_size, usage)

    # Return options for audio processing
    return input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores


def check_options(input_dir1, input_dir2, output_dir, chunk_size, usage):
    """
        Make sure the supplied options are meaningful. Separate from the 
        process_options function to curb function length.
    """

    # Check all arguments before printing usage message
    is_invalid = False

    # Check that input directories exist
    if not isdir(input_dir1):
        print("Input directory 1 does not exist")
        is_invalid = True

    if not isdir(input_dir2):
        print("Input directory 2 does not exist")
        is_invalid = True

    # Check that there are indeed wav files in input dirs 1 and 2
    for f in os.listdir(input_dir1):
        hdr = sndhdr.what(join(input_dir1, f))
        if hdr is not None:
            if hdr[0] == 'wav':
                break
    else:
        print("No wavs in input directory 1")
        is_invalid = True

    for f in os.listdir(input_dir2):
        hdr = sndhdr.what(join(input_dir2, f))
        if hdr is not None:
            if hdr[0] == 'wav':
                break
    else:
        print("No wavs in input directory 2")
        is_invalid = True

    # Check if output directory exists. If it doesn't try to make the dir tree
    if not isdir(output_dir):
        try:
            os.mkdirs(output_dir)
        except:
            traceback.print_exc(file=sys.stdout)
            is_invalid = True

    # Verify that chunk size is between 10 and 1000
    if isinstance(chunk_size, int):
        if (chunk_size < 10) or (chunk_size > 1000):
            print("Chunk size must be between 10 and 1000 (milliseconds)")
            is_invalid = True
    else:
        print("Chunk size must be an integer between 10 and 1000")
        is_invalid = True
            
    # If problem(s) with arguments, print usage and exit
    if is_invalid:
        print(usage)
        sys.exit(1)


def load_wav(wave_filepath):
    """ 
        Convenience function to load the wav file but also to get all the 
        additional data into another variable to use later

        Input: Filepath to wav 

        Output: Complete wave_read object and the named tuple with params
    """

    wav = wave.open(wave_filepath, 'rb')
    params = wav.getparams()
    return wav, params


def main():
    """
        TODO: Write this docstring
    """
    input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores = process_options()

    for f in os.listdir(input_dir1):
        wv_hdr = sndhdr.what(join(input_dir1, f))
        if wv_hdr is not None:
            if wv_hdr[0] == 'wav':
                print('File is a wav, printing hdr')
                print(wv_hdr)
                wv, wv_params = load_wav(join(input_dir1, f))
                print('Printing wav params')
                print(wv_params)
            else:
                continue


    pass


if __name__ == "__main__":
    main()