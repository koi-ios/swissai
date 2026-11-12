i=0
while true; do i=$((i+1)); echo "$i" >> "/root/simple-serving-stack/experiments/checkpoint-restore-exp/results/gate1_criu_cpu/criu-cpu-run/tick.txt"; sleep 0.5; done
