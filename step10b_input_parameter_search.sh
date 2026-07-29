#!/bin/bash
# chmod u+x step10b_input.sh

target_directory='parameter_runs/'

run_index=601

for k_s_i in {0.00005,0.0001}  #{0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01}
do
  for m_basal_i in {0.001,0.02,0.05}
  do
    echo "k_s = $k_s_i, m_basal=$m_basal_i, run_index=$run_index"
    # make a new directory
    new_dir=$target_directory'run_'$run_index/
    mkdir -p $new_dir

    # parameters
    e_v=0.0001
    l_r=5
    h_r=1.0
    k_s=$k_s_i
    k_c=0.1
    l_c=10
    beta2=3.7e5
    p=1.2
    q=1.0
    transition=false
    alpha_s=1.25
    beta_s=1.5
    omega=0.001
    As_factor=5
    sig_topo=5
    melt_input='MAR'
    m_basal=$m_basal_i
    moulins=false
    t_end=10

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

    # run the model
    options="--t_end $t_end --e_v $e_v --l_r $l_r --h_r $h_r --k_s $k_s --k_c $k_c --l_c $l_c --beta2 $beta2 --p $p --q $q --run_index $run_index --alpha_s $alpha_s --beta_s $beta_s --omega $omega --As_factor $As_factor --sig_topo $sig_topo --m_basal $m_basal --melt_input $melt_input"
    if [ "$transition" = true ]; then
      options="$options --transition"
    fi
    if [ "$moulins" = true ]; then
      options="$options --moulins"
    fi
    nohup python -u step10b_russel_coupled.py $options > parameter_runs/log_run_$run_index.txt &

    # increase run_index by 1
    ((run_index=run_index+1))
  done
done
