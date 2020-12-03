import re
import gym
import os

def atari_games():
    pattern = re.compile('\w+NoFrameskip-v4')
    return [env_spec.id for env_spec in gym.envs.registry.all() if pattern.match(env_spec.id)]

env_names = atari_games()
env_names.remove('QbertNoFrameskip-v4')
env_names.remove('ElevatorActionNoFrameskip-v4')
env_names.remove('DefenderNoFrameskip-v4')
num_ales_list = [1024, 2048, 16, 4096] #[1, 32, 64, 128, 256, 512, 1024, 2048, 4096]

for num_ales in num_ales_list:
    for env_name in env_names:

        if num_ales < 1025:
            os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize --use-openai')
        os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize')
        os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize --use-cuda-env')
