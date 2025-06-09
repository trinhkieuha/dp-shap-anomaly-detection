#!/bin/bash
python3.11 "scripts/dpsgd_model.py" --version 202506070329 --continue_run True --n_calls=70 #AUC-------5
python3.11 "scripts/dpsgd_model.py" --version 202506070611 --continue_run True --n_calls=70 #AUC-------3
python3.11 "scripts/dpsgd_model.py" --version 202506071334 --continue_run True --n_calls=70 #AUC-------1
python3.11 "scripts/dpsgd_model.py" --version 202506080706 --continue_run True --n_calls=50 #F1--------5
python3.11 "scripts/dpsgd_model.py" --version 202506081049 --continue_run True --n_calls=50 #F1--------3
python3.11 "scripts/dpsgd_model.py" --version 202506081557 --continue_run True --n_calls=30 #F1--------1
python3.11 "scripts/dpsgd_model.py" --version 202506081954 --continue_run True --n_calls=50 #Recall----5
python3.11 "scripts/dpsgd_model.py" --version 202506082214 --continue_run True --n_calls=50 #Recall----3
python3.11 "scripts/dpsgd_model.py" --version 202505171800 --continue_run True --n_calls=30 #Recall----1
python3.11 "scripts/dpsgd_model.py" --version 202505172146 --continue_run True --n_calls=30 #Precision-5
python3.11 "scripts/dpsgd_model.py" --version 202505180036 --continue_run True --n_calls=50 #Precision-3
python3.11 "scripts/dpsgd_model.py" --version 202505180036 --continue_run True --n_calls=30 #Precision-1