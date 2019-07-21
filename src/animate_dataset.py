from simulator.sokoban_world import SokobanWorld
import console_utils
import numpy as np
import settings
import argparse
import utils
import time
import sys
import os

if sys.version_info[0] == 2:
    input = raw_input


# Parse arguments
parser = argparse.ArgumentParser(description='Generate data.')
parser.add_argument('dataset_name', type=str,
                    help='Name of the dataset you want to view.')
parser.add_argument('-r', '--randomize', action='store_true',
                    help='Randomize all samples.')
parser.add_argument('-g', '--grouped', action='store_true',
                    help='Randomize instances only (keep variations grouped).')
parser.add_argument('-s', '--screenshot', action='store_true',
                    help='Enable screenshot mode.')
args = parser.parse_args()


# Define params
PAUSE_TIME = 5.1 # time between time-steps, in seconds


# Auto generated params:
TEMP_DIR = "/tmp"

# Load data and initialize grid world
f = utils.load_dataset(args.dataset_name, TEMP_DIR)
X, Y = f.attrs["world_shape"]
env = SokobanWorld()
env.render_MPL()


try:
    # Print dataset info
    print ("")
    print ("Description: {}".format(f.attrs["description"]))
    print ("World Dims: {}".format(f.attrs["world_shape"]))
    print ("Number of Objects: {}".format(f.attrs["num_objects"]))
    print ("Number of Valid Actions: {}".format(f.attrs["num_actions"]))
    print ("")

    

    # Animate data set
    num_actions = f.attrs["num_actions"]
    indices = range(f.attrs["num_samples"])

    t_p = 0.9
    train_split = t_p*f.attrs["num_samples"]
    test_split = train_split+f.attrs["num_samples"]*(1.0-t_p)/2
    train_index = np.linspace(0,int(train_split),int(train_split),dtype=int)
    test_index = indices[int(train_split):int(test_split)]
    val_index = indices[int(test_split):]

    if args.grouped:
        indices = np.reshape(indices, [f.attrs["num_env_instances"], f.attrs["num_variations_per_instance"]])
    if args.randomize:
        np.random.shuffle(indices)
    if args.grouped:
        indices = indices.flatten()
    for i in test_index[40:50]:
        print ("Executing {}/{}".format(i,f.attrs["num_samples"]))
        T =  f["sequence_length"][i]
        env.reset_world(f["wall"][i], f["robot_loc"][i,0], f["obj_loc"][i,:,0], f["goal_loc"][i])

        idx0, idx1 = range(T), f["action"][i,:T]
        action_one_hot = np.zeros([T, num_actions], dtype='bool')
        action_one_hot[idx0, idx1] = 1
        if args.screenshot:
            screenshot = console_utils.query_yes_no("Save screenshot of current environment?", default="no")
            if screenshot:
                filename = input("Inpute a filename: ")
                env.screenshot(settings.PROJECT_PATH+"/data/"+filename)
        time.sleep(PAUSE_TIME)
        for t in range(T):
            env.step(action_one_hot[t])
            time.sleep(0.1)
finally:
    filepath = f.filename
    f.close()
    if filepath.startswith(TEMP_DIR):
        os.remove(filepath)
