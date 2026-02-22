# Description: Script to find the best agent (based on highest episode number)
# from a experiment results stored on the server.

# takes a server name (defaults: tcml1)
# finds all the SACx folders in the home directory of the server
# loops through all the folders in results/ subfolder-folder
# creates a yaml file containing the best agent for each experiment
# assumption: experiment_name is unique across all SACx folders
# copies server:SACx/results/experiment_name/models/episode_XXXX to local results/experiment_name/models/episode_XXXX
# output yaml example:
#   experiment_1:
#     folder_path: results/experiment_1/models/episode_1000
#   experiment_2:
#     folder_path: results/experiment_2/models/episode_1800

import os
import yaml
import re
from pathlib import Path


BASE_PATH = Path("src") / "SAC"


def extract_episode_number(folder_name):
    match = re.search(r'episode_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return -1


def find_best_agent_in_server_folder(server, sac_folder, exp_name=None):
    results_path = f"{sac_folder}/results"
    best_agents = {}
    new_agents = set()

    cmd = f'ssh {server} "ls {results_path}"'
    experiment_folders = os.popen(cmd).read().strip().split('\n')

    for experiment in experiment_folders:
        if exp_name and exp_name not in experiment:
            continue
        if experiment in best_agents.keys():
            print(f"Warning: Duplicate experiment name {experiment} found in {sac_folder}, skipping.")
            continue
        experiment_path = f"{results_path}/{experiment}/models"
        cmd = f'ssh {server} "ls {experiment_path}"'
        model_folders = os.popen(cmd).read().strip().split('\n')

        best_episode = -1
        best_folder = None

        for folder in model_folders:
            episode_number = extract_episode_number(folder)
            if episode_number > best_episode:
                best_episode = episode_number
                best_folder = folder

        if best_folder:
            local_path = BASE_PATH / Path(f"results/{experiment}/models/{best_folder}")
            copy = True
            if os.path.exists(local_path):
                if any(local_path.iterdir()):
                    copy = False
            if os.path.exists(local_path.parent):
                print(f"Warning: Experiment {experiment} already exists with lower episode (new = {best_episode}).")
            if copy:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = f'scp -r {server}:{experiment_path}/{best_folder} {local_path}'
                os.system(cmd)
                cmd = f'scp -r {server}:{results_path}/{experiment}/config.yaml {local_path.parent.parent}/config.yaml'
                os.system(cmd)
            new_agents.add(experiment)
            best_agents[experiment] = {
                'type': 'CustomAgent',
                'experiment_name': experiment,
            }
        else:
            print(f"Warning: No valid model folders found for experiment {experiment} in {sac_folder}.")

    result = {'opponents': []}
    for agent in new_agents:
        result['opponents'].append({
            'type': 'CustomAgent',
            'experiment_name': agent,
        })
    return result


def find_best_agents(exp_name='all', server='tcml1'):
    cmd = f'ssh {server} "ls ~ | grep SAC"'
    sac_folders = os.popen(cmd).read().strip().split('\n')
    sac_folders.extend(["src1/src/SAC"])
    all_agents = {'opponents': []}
    if exp_name == 'all':
        for sac_folder in sac_folders:
            best_agents = find_best_agent_in_server_folder(server, sac_folder)
            all_agents['opponents'].extend(best_agents['opponents'])
    else:
        for sac_folder in sac_folders:
            best_agents = find_best_agent_in_server_folder(server, sac_folder,
                                                           exp_name)
            all_agents['opponents'].append(best_agents['opponents'])

    # Write the results to a YAML file
    with open('best_agents2.yaml', 'a') as yaml_file:
        yaml.dump(dict(all_agents), yaml_file)

    print("Best agents information saved to best_agents.yaml")


if __name__ == "__main__":
    import sys
    exp_name = sys.argv[1] if len(sys.argv) > 1 else 'all'
    find_best_agents(exp_name=exp_name)
