#!/bin/bash
# chmod u+x step10b_input.sh

target_directory='parameter_runs/'

# determine the number of already started runs by counting the number of directories, than add 1 to get new run_index
num_directories=$(find "$target_directory" -maxdepth 1 -mindepth 1 -type d -name "run_*" | wc -l)
# run_index=$((num_directories+1))
run_index=478

# make a new directory and copy this file so that the input parameters are recorded at the start of the simulation
new_dir=$target_directory'run_'$run_index/
mkdir -p $new_dir
cp step10b_input.sh $new_dir'input_run'$run_index'.sh'

# parameters
e_v=0.0001
l_r=5
h_r=1.0
k_s=0.0004
k_c=0.1
l_c=10
beta2=3.7e5
p=1.2
q=1.0
transition=false
alpha_s=1.5
beta_s=1.5
omega=0.001
As_factor=6
sig_topo=5
melt_input='MAR'
m_basal=0.02
moulins=false
t_end=12
beta2_inversion=true

# record parameters in a txt file
cat > "$new_dir/parameters.txt" <<EOF
run_index=$run_index
e_v=$e_v
l_r=$l_r
h_r=$h_r
k_s=$k_s
k_c=$k_c
l_c=$l_c
beta2=$beta2
p=$p
q=$q
transition=$transition
alpha_s=$alpha_s
beta_s=$beta_s
omega=$omega
As_factor=$As_factor
sig_topo=$sig_topo
melt_input=$melt_input
m_basal=$m_basal
moulins=$moulins
t_end=$t_end
EOF

# run the model for 5 years, basis for beta2 inversion
t_end_0=5
options="--t_end $t_end_0 --e_v $e_v --l_r $l_r --h_r $h_r --k_s $k_s --k_c $k_c --l_c $l_c --beta2 $beta2 --p $p --q $q --run_index $run_index --alpha_s $alpha_s --beta_s $beta_s --omega $omega --As_factor $As_factor --sig_topo $sig_topo --m_basal $m_basal --melt_input $melt_input"
if [ "$transition" = true ]; then
  options="$options --transition"
fi
if [ "$moulins" = true ]; then
  options="$options --moulins"
fi
python -u step10b_russel_coupled.py $options > parameter_runs/log_run_$run_index.txt

# run the model with inverted beta2, for longer
options="--t_end $t_end --e_v $e_v --l_r $l_r --h_r $h_r --k_s $k_s --k_c $k_c --l_c $l_c --beta2 $beta2 --p $p --q $q --run_index $run_index --alpha_s $alpha_s --beta_s $beta_s --omega $omega --As_factor $As_factor --sig_topo $sig_topo --m_basal $m_basal --melt_input $melt_input"
if [ "$transition" = true ]; then
  options="$options --transition"
fi
if [ "$moulins" = true ]; then
  options="$options --moulins"
fi
if [ "$beta2_inversion" = true ]; then
  options="$options --beta2_inversion"
  python -u step12b_inversion.py $options
  python -u step10b_russel_coupled.py $options > parameter_runs/log_run_$run_index.txt
fi
