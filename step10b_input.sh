#!/bin/bash
# chmod u+x step10b_input.sh

target_directory='parameter_runs/'

# determine the number of already started runs by counting the number of directories, than add 1 to get new run_index
num_directories=$(find "$target_directory" -maxdepth 1 -mindepth 1 -type d -name "run_*" | wc -l)
# run_index=$((num_directories+1))
run_index=124

# make a new directory and copy this file so that the input parameters are recorded at the start of the simulation
new_dir=$target_directory'run_'$run_index/
mkdir $new_dir
cp step10b_input.sh $new_dir'input_run'$run_index'.sh'

# parameters
e_v=0.0001
l_r=5
h_r=1.2
k_s=0.01
k_c=0.5
l_c=10
beta2=5e5
p=1.2
q=0.5
transition=true
alpha_s=1.5
beta_s=1.5
omega=0.001
As_factor=2
sig_topo=5
melt_input='MAR'
moulins=false

# run the model
options="--e_v $e_v --l_r $l_r --h_r $h_r --k_s $k_s --k_c $k_c --l_c $l_c --beta2 $beta2 --p $p --q $q --run_index $run_index --alpha_s $alpha_s --beta_s $beta_s --omega $omega --As_factor $As_factor --sig_topo $sig_topo --melt_input $melt_input"
if [ "$transition" = true ]; then
  options="$options --transition"
fi
if [ "$moulins" = true ]; then
  options="$options --moulins"
fi
python -u step10b_russel_coupled.py $options
