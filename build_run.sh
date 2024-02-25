node_num=$1
if [ -d "build" ]; then
    cd build;
    make;
    mpirun -np $node_num ./example | tee mpi_run.txt;
    cd -
else
    mkdir build;
    cd build;
    cmake ..;
    make;
    mpirun -np $node_num ./example | tee mpi_run.txt;
    cd -
fi
