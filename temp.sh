clear
declare -a lr_arr=(0.001 0.005 0.01 0.05)
declare -a epoch_arr=(1 2 4 8 16 32 64)
declare -a param_arr=('0.01 1 blocked' '0.02 2 interleaved')

for seed in {0..19}; do 
  myvar=`python temp.py ${seed}`
  echo ${myvar}
done