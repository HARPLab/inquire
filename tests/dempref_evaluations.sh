python corl22.py --domain lander --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name no_bias_lander_3_demos
python corl22.py --domain lander --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name no_bias_lander_3_demos_static --static_state

python corl22.py --domain linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name og_lds_3_demos
python corl22.py --domain linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name og_lds_3_demos --static_state

#python corl22.py --domain lander --agent biased_dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lander_3_demos
#python corl22.py --domain pats_linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lds_3_demos
#python corl22.py --domain linear_combo --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name linear_combo_3_demos
#
#python corl22.py --domain lander --agent biased_dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lander_3_demos_static --static_state
#python corl22.py --domain pats_linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lds_3_demos_static --static_state
#python corl22.py --domain linear_combo --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name linear_combo_3_demos_static --static_state
