import argparse
import subprocess
import os

parser = argparse.ArgumentParser(description='Test A2C+V-trace, multiple configurations')
parser.add_argument('--game-name', default='PongNoFrameskip-v4', help='name of the game (default = PongNoFrameskip-v4)')
parser.add_argument('--t-max', default=20, type=int, help='number of training frames (default=20M)')
args = parser.parse_args()

envs = [120, 120, 120, 1200, 1200, 1200, 1200, 1200]
n_steps = [5, 5, 20, 20, 5, 5, 20, 20]
n_steps_per_update = [5, 1, 1, 1, 5, 1, 1, 1]
n_minibatches = [1, 5, 20, 20, 1, 5, 20, 20]
n_gpus = [0, 0, 0, 0, 1, 1, 1, 4]
n_configs = len(n_gpus)

for n_test in range(0, 3):
    for n_config in range(0, n_configs):

        t_max = args.t_max
        if n_gpus[n_config] == 0:
            base_cmd_string = ' ' # --use-openai'
        if n_gpus[n_config] == 1:
            base_cmd_string = ' --use-cuda-env'
        if n_gpus[n_config] == 4:
            t_max = t_max * 4
            base_cmd_string = ' --multiprocessing-distributed --use-cuda-env'
        base_cmd_string = base_cmd_string + ' --normalize ' #--use-openai-test-env'
        output_filename = 'a2cvtrace_' + args.game_name + '_nenvs_' + str(envs[n_config]) + '_nsteps_' + str(n_steps[n_config]) + \
                          '_nstepsperupdate_' + str(n_steps_per_update[n_config]) + '_nminibatches_' + str(n_minibatches[n_config]) + \
                          '_n_gpus_' + str(n_gpus[n_config]) + '_ntest_' + str(n_test) + '.csv'
        common_cmd_string = ' --env-name=' + args.game_name + ' --num-ales=' + str(envs[n_config]) + \
                            ' --num-steps=' + str(n_steps[n_config]) + ' --num-steps-per-update=' + str(n_steps_per_update[n_config]) + \
                            ' --num-minibatches=' + str(n_minibatches[n_config]) + ' --t-max=' + str(t_max) + \
                            ' --evaluation-interval=500000 --output-filename=/results/' + output_filename

        cmd_string = base_cmd_string + common_cmd_string
        os.system('python vtrace_main.py ' + cmd_string)
