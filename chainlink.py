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

    --input2      Directory conatining the "chain links," or a bunch of wavs that
                  the script will use to recreate the wavs in 'input1'

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

    -t            Specifies comparision type (0-2). 0="Pearson", 1="Manhattan",
                  or 2="Mahalanobis". Defaults to "Pearson"

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
        if opt == "-n":
            normalization_type = arg
        if opt == "-t":
            compare_type = arg
        if opt == "-h":
            print(usage)
            sys.exit(0)

    # Make sure that arguments existed for all mandatory options
    if mandatory_checks != 4:
        print(os.linesep + 'Errors detected with mandatory options')
        print(usage)
        sys.exit(1)

    # Verify usability of passed arguments
    check_options(input_dir1, input_dir2, output_dir, chunk_size, compare_type, normalization_type, usage)

    # Return options for audio processing
    return input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores, compare_type, normalization_type


def check_options(input_dir1, input_dir2, output_dir, chunk_size, compare_type, normalization_type, usage):
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
        if (chunk_size < 1) or (chunk_size > 1000):
            print("Chunk size must be between 1 and 1000 (milliseconds)")
            is_invalid = True
    else:
        print("Chunk size must be an integer between 1 and 1000")
        is_invalid = True

    # Check that comparison type is valid
    if compare_type not in ("Pearson", "Manhattan", "Mahalanobis"):
        print('Comparison type (-t) must be either 0, 1, or 2. 0="Pearson", ', 
              '1="Manhattan", and 2="Mahalanobis".')
        is_invalid = True

    # Check that normalization type is valid
    if normalization_type not in ("None", "Max", "Stdev"):
        print('Comparison type (-t) must be either 0, 1, or 2. 0="None", ', 
              '1="Max", and 2="Stdev"')
        is_invalid = True
    
    # If problem(s) with arguments, print usage and exit
    if is_invalid:
        print(usage)
        sys.exit(1)


def get_valid_wavs():
    # TODO
    pass


def load_wav(wave_filepath):
    """ 
        Convenience function to load the wav file but also to get all the 
        additional data into other variables to use later

        Input: Filepath to wav 

        Output: Complete wave_read object and the named tuple with params
    """
    # TODO: Load addition parameters, nchannels and sample width, to help with
    # converting wav files in input dir 2 to have the same parameters as the 
    # wav file from input dir 1 

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
    # TODO
    pass


def match_wav_params():
    """
        Takes two wav files and their parameters and and converts the second 
        wav to have the same parameters as the first.
    """
    # TODO
    pass


def normalize_chunk(chunk1, chunk2, normalization_type):
    """
        Takes a chunk from input wav 1 (the wav being resynthesized) and the
        best-fitting chunk from the set of wavs in input dir 2. Also takes the
        normalization_type parameter set by the user from the command line.

        Roughly match the amplitude of the replacement chunk to the original 
        chunk based on the normalization type.

        If not mono, scale each channel separately.

        If normalization_type == "Max", scale chunk two by the following
        max(chunk1)/max(chunk2)

        If normalization_type == "Stdev", scale chunk two with the following
        y = mean(chunk1) + (x - mean(chunk2)) x stdev(chunk1)/stdev(chunk2),
        where x is original vector for chunk2 and y is the vector after scaling

        return new_chunk2
    """

    if normalization_type == "Max":
        # TODO: Untested
        new_chunk2 = np.zeros(chunk2.shape)
        # Process channels separately. At this point in the script, the 
        # number of channels should be equal for chunks 1 and 2
        for chn in range(chunk2.shape[1]):
            max1 = np.max(chunk1[:, chn])
            max2 = np.max(chunk2[:, chn])
            new_chunk2[:, chn] = chunk2[:, chn] * max1/max2
        
    elif normalization_type == "Stdev":
        # TODO: Untested
        mean1 = np.mean(chunk1)
        mean2 = np.mean(chunk2)
        s1 = np.std(chunk1, ddof=1)
        s2 = np.std(chunk2, ddof=1)
        for chn in range(chunk2.shape[1]):
            new_chunk2[:, chn] = mean1 + (chunk2[:, chn] - mean2) * s1/s2

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


def main():
    """
        TODO: Write this docstring
    """
    input_dir1, input_dir2, output_dir, chunk_size, is_verbose, is_mp, cores = process_options()


    # Store valid wav files for input dirs 1 and 2 into lists to avoid huge 
    # indent blocks from checking file validity before working with them
    input1_wavs = []
    # TODO: Encapsulate this logic into a get_valid_wavs function
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
            # TODO: convert wv2_np to have the same parameters as wv1_np
            wv2, wv_params2, wv_framerate2, wv2_nframes, wv2_np = load_wav(g)
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
                #       at a time to mitigate phasing issues.
                for j in range(nchunks_wv2):
                    wv2_chunk = wv2_np.data[j*chunk_size_frms:j*chunk_size_frms+chunk_size_frms, :]
                    corr_count = 0
                    chunk_corr = 0
                    # TODO: Change this or implement different correlation types.
                    #       It might not make sense to correlate the left channel 
                    #       of one wav with the right channel of another wav
                    #       and then average it with the other 3 correlations 
                    #       (LL, RR, and RL). That's what this code block 
                    #       currently does. Maybe there should be a "full corr"
                    #       option and a "same channel corr" option.
                    # TODO: Also, implement different ways of doing this 
                    #       similarity check. Maybe implement mahalanobis and
                    #       manhattan distance functions. You could stack the 
                    #       channels into a single vector for each wav and then
                    #       compare the vectors with distance functions. Choose 
                    #       the best chunk based on shortest distance...
                    # TODO: Implement a similarity metric that takes advantage 
                    #       of FFT and looks at similarity across maybe 10 
                    #       frequency bins.
                    for wv1_channel in range(wv1_chunk.shape[1]):
                        for wv2_channel in range(wv2_chunk.shape[1]):
                            corr_count += 1
                            channel_corr = np.corrcoef(wv1_chunk[:, wv1_channel], 
                                                        wv2_chunk[:, wv2_channel])[0,1]
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