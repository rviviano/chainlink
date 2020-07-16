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
import scipy.signal
import multiprocessing as mp
from os.path import isdir, isfile, abspath, join, basename, splitext

# Meh, updated whenever I feel like it
__version__ = "0.1.1"

# TODO: Implement a more standardized test suite

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

    --input2      Directory conatining the "chain links," or a bunch of wavs 
                  that the script will use to recreate the wavs in 'input1'

    --output      Directory where you want the script to save output

    --chunk_size  Number between 1 and 1000. The chunk size in milleseconds, 
                  where a chunk is the segment of a sample from input1 that gets
                  replaced by a segment of the same size from a sample within 
                  the input2 directory

    Optional Options:

    -v            Turn verbosity on - increases text output the script generates

    -m            Turn multiprocessing on - leverages multicore systems

    -c            Number of cores to use, defaults to 2 if multiprocessing is 
                  specified but the user doesn't pass an argument to this option

    -n            Normalization method (0-2): 0="None", 1="Max", 2="Stdev". 
                  Defaults to 1 ("Max").

    -t            Specifies comparision type (0-3). 0="Pearson", 1="Manhattan",
                  2="Mahalanobis", or 3="Spectral". Defaults to "Pearson"

    -h            Print this usage message and exit
    """

    # Set verbosity to false
    is_verbose = False

    # Set multiprocessing to false
    is_mp = False

    # Set comparison type to Pearson correlation
    compare_type = "Pearson"

    # Set normalization method to "None" (Not None type).
    normalization_type = "Max"

    # Specify if full-correlation or like-channel correlation
    is_full_correlation = False

    # Set number of cores to use for multiprocessing to 2 as a default
    cores = 2

    # Checks that mandatory options provided. This variable should equal 4 
    # before continuing execution of the script
    mandatory_checks = 0

    # Get commandline options and arguments
    options, _ = getopt.getopt(sys.argv[1:], "hvmc:", ["input1=", "input2=", 
                               "output=", "chunk_size=", "n=", "t="])

    for opt, arg in options:
        # Mandatory arguments
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
        # Optional options
        if opt == "-v":
            is_verbose = True
        if opt == "-m":
            is_mp = True
        if opt == "-c":
            cores = arg
        # Set normalization type
        if opt == "-n":
            if arg == 0:
                normalization_type = "None"
            elif arg == 1:
                normalization_type = "Max"
            elif arg == 2:
                normalization_type = "Stdev"
            else:
                print("Invalid normalization type, ", arg, 
                      ". Defaulting to 'Max'")
        # Set comparison type
        if opt == "-t":
            if arg == 0:
                compare_type = "Pearson"
            elif arg == 1:
                compare_type = "Manhattan"
            elif arg == 2:
                compare_type = "Mahalanobis"
            elif arg == 3:
                compare_type = "Spectral"
            else:
                print("Invalid comparison type, ", arg, 
                      ". Defaulting to 'Pearson'")
        if opt == "-f":
            is_full_correlation = True
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
    return input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores, compare_type, normalization_type, is_full_correlation


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
    for d in enumerate((input_dir1, input_dir2)):
        for f in os.listdir(d[1]):
            hdr = sndhdr.what(join(d[1], f))
            if hdr is not None:
                if hdr[0] == 'wav':
                    break
        else:
            print("No wavs in input directory " + str(d[0]+1))
            is_invalid = True

    # Check if output directory exists. If it doesn't try to make the dir tree
    if not isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            traceback.print_exc(file=sys.stdout)
            is_invalid = True

    # Verify that chunk size is between 1 and 1000
    if isinstance(chunk_size, int):
        if (chunk_size < 1) or (chunk_size > 1000):
            print("Chunk size must be between 1 and 1000 (milliseconds)")
            is_invalid = True
    else:
        print("Chunk size must be an integer between 1 and 1000")
        is_invalid = True
    
    # If problem(s) with arguments, print usage and exit
    if is_invalid:
        print(usage)
        sys.exit(1)


def get_valid_wavs(input_dir):
    """ Scan directory for wav files, store in list, then return the list. 
        At the momemt (07/2020) I am fairly certain that sndhdr skips 32bit
        float wavs"""
    valid_wavs = []

    for f in os.listdir(input_dir):
        wv_hdr = sndhdr.what(join(input_dir, f))
        if wv_hdr is not None:
            if wv_hdr[0] == 'wav':
                valid_wavs.append(join(input_dir, f))

    return valid_wavs


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
    # 16 bit and 32 bit audio, pain for 24 bit audio. This is why
    # the script is dependent on the wavio package  
    wav_np = wavio.read(wave_filepath)                                           

    return wav, params, framerate, nframes, wav_np


def write_wav():
    # TODO: But this might be unnecessary
    pass


def match_wav_params(wv1_np, wv2_np, params1, params2):
    """
        Takes a wav file, its parameters, and the parameters to convert to.
    """
    # First check that sample rates are consistent, if not, resample wv2

    # Then, check if number of channels are correct. If channel count differs,
    # then take the average of wv2's channels and populate the missing channels 
    # with that signal (in the case of scaling up to more channels). If merging
    # to mono, just use the average as the one channel.

    # Make sure that the datatypes match between arrays
    print(params1)
    print(params2)
    print(type(wv2_np))
    print(type(wv2_np.data))
    print(wv2_np.data.shape)
    print(wv1_np)
    print(wv2_np)
    print(wv2_np.data[:10,:])

    print(params1.sampwidth)
    new_wav_data = np.zeros((params2.nframes, params1.nchannels))
    print(new_wav_data.shape)
    # Samplewidth of 3 == samplewidth of 4. Numpy arrays don't support 24bit
    # so the 24-bit int wavs are cast to 32-bit int arrays.
    
    # If downgrading to a smaller sample width, simply truncate. Concatenative
    # synthesis is going to sound glitchy anyway so any noise introduced by 
    # truncation will go with the aesthetic
    sys.exit()
    # TODO

    pass


def normalize_chunk(chunk1, chunk2, normalization_type):
    """
        Takes a chunk from input wav 1 (the wav being resynthesized) and the
        best-fitting chunk from the set of wavs in input dir 2. Also takes the
        normalization_type parameter set by the user from the command line.

        Exactly or roughly match the amplitude of the replacement chunk to the 
        original chunk based on the normalization type.

        Scale each channel separately with vectorized code.

        If normalization_type == "Max", scale chunk two by the following
        max(chunk1)/max(chunk2). This normalization exactly matches the peak
        amplitudes.

        If normalization_type == "Stdev", scale chunk2 with the following:
        y = mean(chunk1) + (chunk2 - mean(chunk2)) * stdev(chunk1)/stdev(chunk2)
        This normalization matches amplitude means and variabilities.

        return new_chunk2
    """

    if normalization_type == "Max":
        new_chunk2 = np.multiply(chunk2, np.max(chunk1, axis=0)/np.max(chunk2, axis=0))
            
    elif normalization_type == "Stdev":
        stdev_ratio = np.std(chunk1, axis=0, ddof=1)/np.std(chunk2, axis=0, ddof=1)     
        new_chunk2 = chunk2 - np.tile(np.mean(chunk2, axis=0), (chunk2.shape[0],1))
        new_chunk2 = np.multiply(new_chunk2, stdev_ratio)
        new_chunk2 += np.tile(np.mean(chunk1, axis=0), (chunk1.shape[0],1)) 

    # Make sure the new array has the same dtype as the original. This will lead 
    # to minor information loss as 64-bit may downcast to 32-bit at best.
    new_chunk2 = new_chunk2.astype(chunk2.dtype, casting='unsafe')      

    return new_chunk2
        

def process_wav():
    # TODO
    pass


def smooth_data():
    # TODO
    pass


def convert_ms_to_frames(chunk_size_ms, framerate):
    """ 
        Convert chunk size in milleconds to chunk size in number of frames
    
        Framerate is in hz (cycles per second), chunk_size is in ms

        So we need to multiply framerate by chunk_size_ms/1000 to get the
        chunk size in frames. Round down to nearest int.
    """
    return int(framerate * (chunk_size_ms / 1000.0))


def declick_wav():
    pass
    

def main():
    """
        TODO: Write this docstring
    """
    input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores, compare_type, normalization_type, is_full_correlation = process_options()


    # Store valid wav files for input dirs 1 and 2 into lists to avoid huge 
    # indent blocks from checking file validity before working with them
    input1_wavs = get_valid_wavs(input_dir1)
    input2_wavs = get_valid_wavs(input_dir2)

    for f in input1_wavs:
        # TODO: Encapsulate the code within this for loop into a process_wav function
        # Define output filename
        output_file = join(output_dir, splitext(basename(f))[0] + "_chainlinked.wav")
        if is_verbose:
            print('\nInput: ', join(f))
            print('Output: ', output_file)
            print('Analyzing file for resynthesis...')
        # Load the input wavfile to be recreated with the chain links of other wavs
        wv, wv_params, wv_framerate, wv_nframes, wv_np = load_wav(f)
        # Get sample width (number of bytes per frame)
        wv_samplewidth = wv.getsampwidth()
        # Convert chunk size from milliseconds to number of frames
        chunk_size_frms = convert_ms_to_frames(chunk_size, wv_framerate)
        # Get the number of chunks for the input wav file

        # TODO: work with the remainders rather than just leaving that last part of the waveform as 0s
        nchunks_wv1, remainder = divmod(wv_nframes, chunk_size_frms)
        if is_verbose:
            print("\nNumber of chunks to work with: ", nchunks_wv1, "\n")

        # Create array for new audio
        new_wv_np = np.zeros((wv_np.data.shape))

        # Create vector the size nchunks_wv1 to keep track of how well the chunks
        # placed in new_wv_np compare to the chunks analyzed when a new file is opened
        best_corr = np.zeros((nchunks_wv1, 1))
        
        # Go through the chunks and find chain links from the wavs in input2
        # that correlate well with the chunks from audio 1, try to recreate 
        # audio 1. Go file by file for the wavs in input_dir2, if a chunk of one
        # file has a better correlation than a chunk previously placed in 
        # new_wv_np, replace the chunk and update the correlation/distance value 
        # at the corresponding location in the best corr vector.

        # Compare all chunks from wav1 with all chunks from wav2, one file at a time
        for g in input2_wavs:
            if is_verbose:
                print('Analyzing: ', g)
            # Load wav file to compare wv1 against
            wv2, wv2_params, wv2_framerate, wv2_nframes, wv2_np = load_wav(g)
            # TODO: convert wv2_np to have the same parameters as wv1_np
            wv2_np = match_wav_params(wv_np, wv2_np, wv_params, wv2_params)
            # TODO: Write convert wav function that takes current wav parameters
            #       (e.g. bit depth sample rate, nchannels, and converts
            nchunks_wv2, _ = divmod(wv2_nframes, chunk_size_frms)
            for i in range(nchunks_wv1):
                if is_verbose:
                    print('Working with chunk ', i+1)
                wv1_chunk = wv_np.data[i*chunk_size_frms:i*chunk_size_frms+chunk_size_frms, :]
                # TODO: Implement slow, medium, and fast modes that move 1/8th, 
                #       1/4, 1/2, or a full chunk at a time. Currently, the code 
                #       moves a full chunk at a time. Default to 1/4 window move 
                #       at a time.
                for j in range(nchunks_wv2):
                    # TODO: Implement different ways of doing this 
                    #       similarity check. Maybe implement mahalanobis and
                    #       manhattan distance functions. You could stack the 
                    #       channels into a single vector for each wav and then
                    #       compare the vectors with distance functions. Choose 
                    #       the best chunk based on shortest distance...
                    # TODO: Implement a similarity metric that takes advantage 
                    #       of FFT and looks at similarity across maybe 10 
                    #       frequency bins.
                    wv2_chunk = wv2_np.data[j*chunk_size_frms:j*chunk_size_frms+chunk_size_frms, :]
                    corr_count = 0
                    chunk_corr = 0

                    if is_full_correlation:
                        # Correlate every channel in wv1 with every channel in wv2
                        # and then take the average of all those correlations
                        for wv1_channel in range(wv1_chunk.shape[1]):
                            for wv2_channel in range(wv2_chunk.shape[1]):
                                corr_count += 1
                                channel_corr = np.corrcoef(wv1_chunk[:, wv1_channel], 
                                                            wv2_chunk[:, wv2_channel])[0,1]
                                if channel_corr == np.nan:
                                    channel_corr = 0

                                chunk_corr += channel_corr
                    else:
                        # Only correlate like-channels and take the average of the
                        # correlations. Assume that the number of channels in wv2
                        # are equivalent to the number of channels in wv1
                        # TODO: enforce this assumption
                        for chnl in range(wv1_chunk.shape[1]):
                            corr_count += 1
                            channel_corr = np.corrcoef(wv1_chunk[:, chnl], wv2_chunk[:, chnl])[0,1]
                            if channel_corr == np.nan:
                                channel_corr = 0

                            chunk_corr += channel_corr
                    chunk_corr = chunk_corr/corr_count
                    if chunk_corr > best_corr[i, 0]:
                        new_wv_np[i*chunk_size_frms:i*chunk_size_frms+chunk_size_frms, :] = np.copy(wv2_chunk)
                        best_corr[i, 0] = chunk_corr
            
        # Write output to a wav file
        wavio.write(output_file, new_wv_np, rate=wv_framerate, sampwidth=wv_samplewidth)


if __name__ == "__main__":
    main()