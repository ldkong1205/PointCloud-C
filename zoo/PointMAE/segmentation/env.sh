export PATH="/mnt/lustre/share/cuda-10.0/bin:/mnt/lustre/share/gcc/gcc-5.3.0/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.0/lib64:/mnt/lustre/share/gcc/mpc-0.8.1/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/jwren/anaconda3/lib:$LD_LIBRARY_PATH"

export CC=/mnt/lustre/share/gcc/gcc-5.3.0/bin/gcc
export CXX=/mnt/lustre/share/gcc/gcc-5.3.0/bin/c++

conda activate point-mae