#!/bin/bash
#export CARLA_ROOT=$PYTHONPATH:/home/fyc/project/carla/
export CARLA_ROOT=$PYTHONPATH:/home/kemove/carla_0.9.10.1/CARLA_0.9.10/
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True


# PolarPointBEV evaluation
export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid.xml
export TEAM_AGENT=team_code/xplan_agent.py
#export TEAM_CONFIG=path/to/weight/XPlan-N.ckpt
export TEAM_CONFIG=/home/kemove/liulei/local_to_kemove/PolarPoint-BEV/PolarPoint-BEV/log/XPlan/epoch=15-val_loss=0.665.ckpt
#export CHECKPOINT_ENDPOINT=path/to/save/jsonfile.json
export CHECKPOINT_ENDPOINT=/home/kemove/liulei/local_to_kemove/PolarPoint-BEV/PolarPoint-BEV/test_liulei/json/jsonfile.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
#export SAVE_PATH=path/to/savedir/
export SAVE_PATH=/home/kemove/liulei/local_to_kemove/PolarPoint-BEV/PolarPoint-BEV/test_liulei/savedir/


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}


