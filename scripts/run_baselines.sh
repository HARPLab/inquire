#/bin/bash
domain=$1
#alphas="0.005 0.01 0.05 0.1"
demo_beta=20.0
pref_beta=20.0
corr_beta=20.0
bnry_beta=20.0
if [ "$domain" != "linear_combo" ]; then
    cache="--use_cache"
else
    cache=""
fi
if [ "$domain" == "pizza" ]; then
    alphas="0.001"
else
    alphas="0.005"
fi

for j in $alphas; do
    if [ "$domain" != "linear_combo" ]; then
        name="demo--${domain}_alpha-${j}"
        echo $name
        python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $demo_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent demo-only
    fi
    name="pref--${domain}_alpha-${j}"
    echo $name
    python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $pref_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent pref-only
    name="corr--${domain}_alpha-${j}"
    echo $name
    python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $corr_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent corr-only
    name="bnry--${domain}_alpha-${j}"
    echo $name
    python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $bnry_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent binary-only
    for n in "0 1 2"; do
        name="dempref--${domain}_${n}_demos"
        echo $name
        python scripts/corl22.py --actual_queries 1 --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --seed_with_n_demos ${n} --output_name $name --agent dempref
    done

    if [ "$domain" != "linear_combo" ]; then
        name="demo--static_${domain}_alpha-${j}"
        echo $name
        python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $demo_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent demo-only --static_state
        name="pref--static_${domain}_alpha-${j}"
        echo $name
        python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $pref_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent pref-only --static_state
        name="corr--static_${domain}_alpha-${j}"
        echo $name
        python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $corr_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent corr-only --static_state
        name="bnry--static_${domain}_alpha-${j}"
        echo $name
        python scripts/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $bnry_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent binary-only --static_state
        for n in "0 1 2"; do
            name="dempref--static_${domain}_${n}_demos"
            echo $name
            python scripts/corl22.py --actual_queries 1 --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --seed_with_n_demos ${n} --output_name $name --agent dempref --static_state
        done
    fi
done
