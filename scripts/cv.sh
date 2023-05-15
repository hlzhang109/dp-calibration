
declare -a train_datasets=('Food101' 'SUN397' 'cifar10')
declare -a epsilons=(100000)
data_root='' 

for data_name in "${train_datasets[@]}";do
    for eps in "${epsilons[@]}";do
        export PYTHONPATH="."
        folder=dp 
        log="cv/results/${folder}/logs/${data_name}_nondp.log"
        cmd="python cv/main_cv.py --no_private True --max_grad_norm 1 --data_root ${data_root} --data_name ${data_name} --target_epsilon ${eps} > ${log} 2>&1 &"  
        ((i++))
        CMD="${cmd}"
        eval ${CMD}
    done
done