#export LD_LIBRARY_PATH="~:$LD_LIBRARY_PATH"
setenv LD_LIBRARY_PATH "~:$LD_LIBRARY_PATH"
setenv LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:~"

setenv PATH "${PATH}:~"


source /CMC/scripts/kit.gpdk45_OA.csh
source /CMC/scripts/cadence.spectre20.10.155.csh
