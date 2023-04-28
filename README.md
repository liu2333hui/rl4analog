# rl4analog

RL4Analog is a light-weight wrapper of an RL optimizer over the Spectre Analog simulator. It assumes that Spectre is installed and that the PDKs are also in the correct path.

There are a number of benchmark analog circuits available to optimize for using RL. Circuit1 is a 2 stage voltage amplifier, and TIA is a two-stage transimpedance amplifier.


To optimize, first source the technology and Spectre requirements (may need to enter the correct tcsh environment), "source run.sh", then run "python train.golden.py", the resuls will be printed into a results.log file.
