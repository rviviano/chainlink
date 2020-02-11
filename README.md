# ChainLink
 Concatenative Synthesis with Python

## Dependencies 
numpy, wavio

## Important Notess:
This software is very early days and therefore probably very buggy. It doesn't
assume a certain sample rate, but it does kinda assume that all the input files
have the same parameters. Furthermore, it works if all the input wavs are 
1-channel or if all the input wavs are 2-channel, but I'm fairly certain the 
script will break if you try to feed it 1-channel and 2-channel input at the 
same time. Finally, do not feed it 32-bit float files. 32-bit int and lower are 
fine. This is a limitation of the wave library. I'll look into alternative
options soon.

 Usage: 
 
    python chainlink.py --input1 <arg> --input2 <arg> --output <arg> 
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

    Optional Options: (Aside from 'help' these are not yet implemented)

    -v            Turn verbosity on - increases text output the script generates

    -m            Turn multiprocessing on - leverages multicore systems

    -c            Number of cores to use, defaults to 2 if multiprocessing is 
                  specified but the user doesn't pass an argument to this option

    -h            Print this usage message and exit
