#/bin/bash
domain=$1
alphas="0.005 0.01 0.05 0.1"
if [ "$domain" != "linear_combo" ]; then
    cache="--use_cache"
else
    cache=""
fi
for j in $alphas; do
    name="demo--${domain}_alpha-${j}"
    echo $name
    python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 50.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent demo-only
    name="pref--${domain}_alpha-${j}"
    echo $name
    python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 20.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent pref-only
    name="corr--${domain}_alpha-${j}"
    echo $name
    python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 20.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent corr-only
    name="bnry--${domain}_alpha-${j}"
    echo $name
    python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 5.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent binary-only
    if [ "$domain" != "linear_combo" ]; then
        name="demo--static_${domain}_alpha-${j}"
        echo $name
        python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 50.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent demo-only --static_state
        name="pref--static_${domain}_alpha-${j}"
        echo $name
        python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 20.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent pref-only --static_state
        name="corr--static_${domain}_alpha-${j}"
        echo $name
        python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 20.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent corr-only --static_state
        name="bnry--static_${domain}_alpha-${j}"
        echo $name
        python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta 5.0 --alpha $j --convergence_threshold 0.01 --output_name $name --agent binary-only --static_state
    fi
done
