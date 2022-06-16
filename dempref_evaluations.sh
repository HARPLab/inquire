python tests/corl22.py --actual_queries 1 --domain linear_system --use_cache --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_system_0_demos  -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_system --use_cache --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_system_1_demos  -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_system --use_cache --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_system_2_demos  -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_system --use_cache --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_system_0_demos_static -I 1000 -N 1000 --static
python tests/corl22.py --actual_queries 1 --domain linear_system --use_cache --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_system_1_demos_static -I 1000 -N 1000 --static
python tests/corl22.py --actual_queries 1 --domain linear_system --use_cache --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_system_2_demos_static -I 1000 -N 1000 --static

python tests/corl22.py --actual_queries 1 --domain lander --use_cache --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--lander_0_demos -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain lander --use_cache --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--lander_1_demos -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain lander --use_cache --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--lander_2_demos -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain lander --use_cache --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--lander_0_demos_static --static -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain lander --use_cache --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--lander_1_demos_static --static -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain lander --use_cache --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--lander_2_demos_static --static -I 1000 -N 1000

python tests/corl22.py --actual_queries 1 --domain linear_combo --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_combo_0_demos  -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_combo --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_combo_1_demos  -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_combo --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_combo_2_demos  -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_combo --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_combo_0_demos_static --static -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_combo --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_combo_1_demos_static --static -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain linear_combo --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--linear_combo_2_demos_static --static -I 1000 -N 1000

python tests/corl22.py --actual_queries 1 --domain pizza --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--pizza_0_demos -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain pizza --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--pizza_1_demos -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain pizza --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--pizza_2_demos -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain pizza --agent dempref --seed_with_n_demos 0 --runs 10 --tests 10 --queries 20 -V --output_name dempref--pizza_0_demos_static --static -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain pizza --agent dempref --seed_with_n_demos 1 --runs 10 --tests 10 --queries 20 -V --output_name dempref--pizza_1_demos_static --static -I 1000 -N 1000
python tests/corl22.py --actual_queries 1 --domain pizza --agent dempref --seed_with_n_demos 2 --runs 10 --tests 10 --queries 20 -V --output_name dempref--pizza_2_demos_static --static -I 1000 -N 1000


#python corl22.py --domain linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name og_lds_3_demos
#python corl22.py --domain linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name og_lds_3_demos --static_state

#python corl22.py --domain lander --agent biased_dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lander_3_demos
#python corl22.py --domain pats_linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lds_3_demos
#
#python corl22.py --domain lander --agent biased_dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lander_3_demos_static --static_state
#python corl22.py --domain pats_linear_system --agent dempref --seed_with_n_demos 3 --runs 10 --tests 10 --queries 20 -V --output_name lds_3_demos_static --static_state
