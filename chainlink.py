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
import wave, sndhdr, wavio
import numpy as np 
import multiprocessing as mp
from os.path import isdir, isfile, abspath, join, basename, splitext

__version__ = "0.1.0"

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

    Optional Options:

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

    # Checks that mandatory options provided. This variable should equal 4 
    # before continuing execution of the script
    mandatory_checks = 0

    # Get commandline options and arguments
    options, _ = getopt.getopt(sys.argv[1:], "hvmc:", ["input1=", "input2=", 
                               "output=", "chunk_size="])

    for opt, arg in options:
        if opt == "--input1":
            if arg is not None:
                input_dir1 = arg
                mandatory_checks += 1
        if opt == "--input2": 
            if arg is not None:
                input_dir2 = arg 
                mandatory_checks += 1
        if opt == "--output":
            if arg is not None:
                output_dir = arg
                mandatory_checks += 1 
        if opt == "--chunk_size":
            if arg is not None:
                chunk_size = int(arg)
                mandatory_checks += 1
        if opt == "-v":
            is_verbose = True
        if opt == "-m":
            is_mp = True
        if opt == "-c":
            cores = arg
        if opt == "-h":
            print(usage)
            sys.exit(0)

    # Make sure that arguments existed for all mandatory options
    if mandatory_checks != 4:
        print(os.linesep + 'Errors detected with mandatory options')
        print(usage)
        sys.exit(1)

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
            os.makedirs(output_dir)
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
        additional data into other variables to use later

        Input: Filepath to wav 

        Output: Complete wave_read object and the named tuple with params
    """
    # Open wav_read object and extract useful parameter information
    wav = wave.open(wave_filepath, 'rb')
    params = wav.getparams()
    framerate = wav.getframerate()
    nframes = wav.getnframes()

    # Convert bytes object to numpy array. Relatively straightforward for
    # 16 bit and 32 bit audio, pain in the ass for 24 bit audio. This is why
    # the script is dependent on the wavio package  
    wav_np = wavio.read(wave_filepath)                                           

    return wav, params, framerate, nframes, wav_np


def write_wav():
    pass


def convert_ms_to_frames(chunk_size_ms, framerate):
    """ 
        Convert chunk size in milleconds to chunk size in number of frames
    
        Framerate is in hz (cycles per second), chunk_size is in ms

        So we need to multiply framerate by chunk_size_ms/1000 to get the
        chunk size in frames. Round down to nearest int.
    """
    return int(framerate * (chunk_size_ms / 1000.0))


def main():
    """
        TODO: Write this docstring
    """
    input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores = process_options()


    # Store valid wav files for input dirs 1 and 2 into lists to avoid huge 
    # indent blocks from checking file validity before working with them
    input1_wavs = []
    for f in os.listdir(input_dir1):
        wv_hdr = sndhdr.what(join(input_dir1, f))
        if wv_hdr is not None:
            if wv_hdr[0] == 'wav':
                input1_wavs.append(join(input_dir1, f))

    input2_wavs = []
    for f in os.listdir(input_dir2):
        wv_hdr = sndhdr.what(join(input_dir2, f))
        if wv_hdr is not None:
            if wv_hdr[0] == 'wav':
                input2_wavs.append(join(input_dir2, f))


    for f in input1_wavs:
        # TODO: Encapsulate this in a process_wav function
        # Define output filename
        output_file = join(output_dir, splitext(basename(f))[0] + "_chainlinked.wav")
        print('\nInput: ', join(f))
        print('Output: ', output_file)
        # Load the input wavfile to be recreated with the chain links of other wavs
        wv, wv_params, wv_framerate, wv_nframes, wv_np = load_wav(f)
        # Get sample width (number of bytes per frame)
        wv_samplewidth = wv.getsampwidth()
        # Convert chunk size from milliseconds to number of frames
        chunk_size_frms = convert_ms_to_frames(chunk_size, wv_framerate)
        # Get the number of chunks for the input wav file
        nchunks_wv1, remainder = divmod(wv_nframes, chunk_size_frms)
        print("\nNumber of chunks to work with: ", nchunks_wv1, "\n")
        # Create array for new audio
        new_wv_np = np.zeros((wv_np.data.shape))
        # Go through the chunks and find chain links from the wavs in input2
        # that correlate well with the chunks from audio 1, try to recreate 
        # audio 1
        for i in range(nchunks_wv1):
            print('Working with chunk ', i+1)
            wv1_chunk = wv_np.data[i*chunk_size_frms:i*chunk_size_frms+chunk_size_frms, :]
            best_wv2_chunk = np.zeros(wv1_chunk.shape)
            best_corr = 0
            for g in input2_wavs:
                print('Analyzing: ', g)
                _, _, _, wv2_nframes, wv2_np = load_wav(g)
                nchunks_wv2, remainder2 = divmod(wv2_nframes, chunk_size_frms)
                for j in range(nchunks_wv2):
                    wv2_chunk = wv2_np.data[j*chunk_size_frms:j*chunk_size_frms+chunk_size_frms, :]
                    corr_count = 0
                    chunk_corr = 0
                    for wv1_channel in range(wv1_chunk.shape[1]):
                        for wv2_channel in range(wv2_chunk.shape[1]):
                            corr_count += 1
                            channel_corr = np.corrcoef(wv1_chunk[:, wv1_channel], 
                                                       wv2_chunk[:, wv2_channel])[0,1]
                            if channel_corr == np.nan:
                                channel_corr = 0

                            chunk_corr += channel_corr
                    chunk_corr = chunk_corr/corr_count
                    if chunk_corr > best_corr:
                        best_wv2_chunk = np.copy(wv2_chunk)
                        best_corr = chunk_corr
            
            new_wv_np[i*chunk_size_frms:i*chunk_size_frms+chunk_size_frms, :] = np.copy(best_wv2_chunk)
        # Write output to a wav file
        wavio.write(output_file, new_wv_np, rate=wv_framerate, sampwidth=wv_samplewidth)



if __name__ == "__main__":
    main()