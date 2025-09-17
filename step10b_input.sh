#!/bin/bash
# chmod u+x step10b_input.sh

target_directory='parameter_runs/'

# determine the number of already started runs by counting the number of directories, than add 1 to get new run_index
num_directories=$(find "$target_directory" -maxdepth 1 -mindepth 1 -type d -name "run_*" | wc -l)
run_index=$((num_directories+1))

# make a new directory and copy this file so that the input parameters are recorded at the start of the simulation
new_dir=$target_directory'run_'$run_index/
mkdir $new_dir
cp step10b_input.sh $new_dir'input_run'$run_index'.sh'

# parameters
e_v=0.0001
l_r=5
h_r=0.5
k_s=0.05
k_c=0.5
l_c=10
beta2=1e6
p=1.0
q=0.5

# run the model
python step10b_russel_coupled.py --englacial_void_ratio $e_v --bump_spacing $l_r --bump_height $h_r --sheet_conductivity $k_s --channel_conductivity $k_c --sheet_width_below_channel $l_c --basal_traction $beta2 --pressure_exponent $p --sliding_exponent $q --run_index $run_index
