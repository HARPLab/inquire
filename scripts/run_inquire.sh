#/bin/bash
domain=$1
#alphas="0.005 0.01 0.05 0.1"
if [ "$domain" != "linear_combo" ]; then
    cache="--use_cache"
    agent="inquire"
else
    cache=""
    agent="no-demos"
fi
if [ "$domain" == "pizza" ]; then
    alphas="0.001"
else
    alphas="0.005"
fi
demo_beta=20.0
pref_beta=20.0
corr_beta=20.0
bnry_beta=20.0
conv=0.01
for a in $alphas; do
    name="inquire--${domain}_alpha-${a}"
    echo $name
    python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --alpha $a --convergence_threshold $conv --output_name $name --betas "{Modality.DEMONSTRATION: $demo_beta, Modality.PREFERENCE: $pref_beta, Modality.CORRECTION: $corr_beta, Modality.BINARY: $bnry_beta}" --agent ${agent} 
    if [ "$domain" != "linear_combo" ]; then
        name="inquire--static_${domain}_alpha-${a}"
        echo $name
        python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --alpha $a --convergence_threshold $conv --output_name $name --betas "{Modality.DEMONSTRATION: $demo_beta, Modality.PREFERENCE: $pref_beta, Modality.CORRECTION: $corr_beta, Modality.BINARY: $bnry_beta}" --static_state --agent ${agent}
    fi
done
