#/bin/bash
domain="pizza"
alphas="0.0005"
demo_beta=20.0
pref_beta=20.0
corr_beta=20.0
bnry_beta=20.0
cache=""
for j in $alphas; do
    #name="pref--${domain}_alpha-${j}"
    #echo $name
    #python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $pref_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent pref-only -V
    #name="corr--${domain}_alpha-${j}"
    #echo $name
    #python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $corr_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent corr-only
    #name="bnry--${domain}_alpha-${j}"
    #echo $name
    #python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $bnry_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent binary-only -V
    if [ "$domain" != "linear_combo" ]; then
        #name="pref--static_${domain}_alpha-${j}"
        #echo $name
        #python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $pref_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent pref-only --static_state -V
        #name="corr--static_${domain}_alpha-${j}"
        #echo $name
        #python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $corr_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent corr-only --static_state
        #name="bnry--static_${domain}_alpha-${j}"
        #echo $name
        #python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $bnry_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent binary-only --static_state -V
        name="demo--static_${domain}_alpha-${j}"
        echo $name
        python tests/corl22.py --domain ${domain} $cache -I 1000 -N 1000 --queries 20 --tests 10 --runs 10 --beta $demo_beta --alpha $j --convergence_threshold 0.01 --output_name $name --agent demo-only --static_state -V
    fi
done
