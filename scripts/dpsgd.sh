#!/bin/bash
#python3.11 "scripts/dpsgd_model.py" --metric auc --n_calls=30 --epsilon 10 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric auc --n_calls=30 --epsilon 5 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric auc --n_calls=30 --epsilon 3 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric auc --n_calls=30 --epsilon 1 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric f1_score --n_calls=30 --epsilon 10 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric f1_score --n_calls=30 --epsilon 5 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric f1_score --n_calls=30 --epsilon 3 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric f1_score --n_calls=30 --epsilon 1 --delta 1e-5 --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric recall --n_calls=30 --epsilon 10 --delta 1e-5 --random_starts=9
python3.11 "scripts/dpsgd_model.py" --metric recall --n_calls=30 --epsilon 5 --delta 1e-5  --random_starts=9
python3.11 "scripts/dpsgd_model.py" --metric recall --n_calls=30 --epsilon 3 --delta 1e-5  --random_starts=9
python3.11 "scripts/dpsgd_model.py" --metric recall --n_calls=30 --epsilon 1 --delta 1e-5  --random_starts=9
#python3.11 "scripts/dpsgd_model.py" --metric precision --n_calls=30 --epsilon 10 --delta 1e-5 --random_starts=9
python3.11 "scripts/dpsgd_model.py" --metric precision --n_calls=30 --epsilon 5 --delta 1e-5  --random_starts=9
python3.11 "scripts/dpsgd_model.py" --metric precision --n_calls=30 --epsilon 3 --delta 1e-5  --random_starts=9
python3.11 "scripts/dpsgd_model.py" --metric precision --n_calls=30 --epsilon 1 --delta 1e-5  --random_starts=9