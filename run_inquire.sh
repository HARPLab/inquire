#/bin/bash
domain=$1
alphas="0.005 0.01 0.05 0.1"
if [ "$domain" != "linear_combo" ]; then
    cache="--use_cache"
    agent="inquire"
else
    cache=""
    agent="no-demos"
fi
i=50.0
j=20.0
k=5.0
conv=0.01
for j in $alphas; do
    name="inquire--${domain}_alpha-${j}"
    echo $name
    python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --alpha $j --convergence_threshold $conv --output_name $name --betas "{Modality.DEMONSTRATION: $i, Modality.PREFERENCE: $j, Modality.CORRECTION: $j, Modality.BINARY: $k}" --agent ${agent}
    if [ "$domain" != "linear_combo" ]; then
        name="inquire--static_${domain}_alpha-${j}"
        echo $name
        python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --alpha $j --convergence_threshold $conv --output_name $name --betas "{Modality.DEMONSTRATION: $i, Modality.PREFERENCE: $j, Modality.CORRECTION: $j, Modality.BINARY: $k}" --static_state --agent ${agent}
    fi
done
