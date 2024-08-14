#!/bin/bash

python run_simulation.py --setting simu1
python run_simulation.py --setting simu2
python run_simulation.py --setting simu3

python hulc.py --setting simu1
python hulc.py --setting simu2
python hulc.py --setting simu3

python subsampling.py --setting simu1
python subsampling.py --setting simu2
python subsampling.py --setting simu3

python bootstrap.py --setting simu1
python bootstrap.py --setting simu2
python bootstrap.py --setting simu3