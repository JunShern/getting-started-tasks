import cv2
import time
import gym
import minerl
import numpy as np
import os
import torch
from pathlib import Path
from torch import nn
from torch._C import LongStorageBase
from tqdm import tqdm

"""
Your task: Implement behavioural cloning for MineRLTreechop-v0.

Behavioural cloning is perhaps the simplest way of using a dataset of demonstrations to train an agent:
learn to predict what actions they would take, and take those actions.
In other machine learning terms, this is almost like building a classifier to classify observations to
different actions, and taking those actions.

For simplicity, we build a limited set of actions ("agent actions"), map dataset actions to these actions
and train on the agent actions. During evaluation, we transform these agent actions (integerse) back into
MineRL actions (dictionaries).

To do this task, fill in the "TODO"s and remove `raise NotImplementedError`s.

Note: For this task you need to download the "MineRLTreechop-v0" dataset. See here:
https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
"""

def render(observation, environment, headless = False, write_frame = False):
    """A function for rendering MineRL environments. You do not need to worry about this function"""
    if isinstance(environment.unwrapped, minerl.env._singleagent._SingleAgentEnv):
        # Environment is a MineRL one, use OpenCV image showing to show image
        # Make it larger for easier reading
        image = observation["pov"]
        image = cv2.resize(image, (256, 256))
        if write_frame:
            video_frames.append(image[..., ::-1])
        if not headless:
            cv2.imshow("minerl-image", image[..., ::-1])
            # Refresh image
            _ = cv2.waitKey(1)
    else:
        # Regular render
        environment.render()

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    Nicked from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# class ConvNet(nn.Module):
#     """
#     :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
#     :param output_dim: Dimensionality of the output vector
#     """

#     def __init__(self, input_shape, output_dim):
#         super().__init__()
#         # TODO Create a torch neural network here to turn images (of shape `input_shape`) into
#         #      a vector of shape `output_dim`. This output_dim matches number of available actions.
#         #      See examples of doing CNN networks here https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn
#         # input_shape = (3, 64, 64)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
#         raise NotImplementedError("TODO implement a simple convolutional neural network here")

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         # TODO with the layers you created in __init__, transform the `observations` (a tensor of shape (B, C, H, W)) to
#         #      a tensor of shape (B, D), where D is the `output_dim`
#         raise NotImplementedError("TODO implement forward function of the neural network")


def agent_action_to_environment(noop_action, agent_action):
    """
    Turn an agent action (an integer) into an environment action.
    This should match `environment_action_batch_to_agent_actions`,
    e.g. if attack=1 action was mapped to agent_action=0, then agent_action=0
    should be mapped back to attack=1.

    noop_action is a MineRL action that does nothing. You may want to
    use this as a template for the action you return.

    0 = forward
    1 = jump
    2 = turn camera left
    3 = turn camera right
    4 = turn camera up
    5 = turn camera down
    6 = attack
    """
    # raise NotImplementedError("TODO implement agent_action_to_environment (see docstring)")
    env_action = noop_action
    if agent_action == 0:
        env_action['forward'] = 1
    elif agent_action == 1:
        env_action['jump'] = 1
    elif agent_action == 2: # TODO(Jun): Make sure the camera works the way I assumed
        env_action['camera'][0] = -45
    elif agent_action == 3:
        env_action['camera'][0] = 45
    elif agent_action == 4:
        env_action['camera'][1] = 45
    elif agent_action == 5:
        env_action['camera'][1] = -45
    elif agent_action == 6:
        env_action['attack'] = 1
    return env_action


def environment_action_batch_to_agent_actions(dataset_actions):
    """
    Turn a batch of actions from environment (from BufferedBatchIterator) to a numpy
    array of agent actions.

    Agent actions _have to_ start from 0 and go up from there!

    For MineRLTreechop, you want to have actions for the following at the very least:
    - Forward movement
    - Jumping
    - Turning camera left, right, up and down
    - Attack

    For example, you could have seven agent actions that mean following:
    0 = forward
    1 = jump
    2 = turn camera left
    3 = turn camera right
    4 = turn camera up
    5 = turn camera down
    6 = attack

    This should match `agent_action_to_environment`, by converting dictionary
    actions into individual integeres.

    If dataset action (dict) does not have a mapping to agent action (int),
    then set it "-1"
    """

    """
    Action space: 
    Dict(
        attack:Discrete(2), 
        back:Discrete(2), 
        camera:Box(low=-180.0, high=180.0, shape=(2,)), 
        craft:Enum(crafting_table,none,planks,stick,torch), 
        equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe), 
        forward:Discrete(2), 
        jump:Discrete(2), 
        left:Discrete(2), 
        nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe), 
        nearbySmelt:Enum(coal,iron_ingot,none), place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch), 
        right:Discrete(2), 
        sneak:Discrete(2), 
        sprint:Discrete(2)
    )
    """

    # There are dummy dimensions of shape one
    batch_size = len(dataset_actions["camera"])
    actions = np.zeros((batch_size,), dtype=np.int)

    # 0 = forward
    # 1 = jump
    # 2 = turn camera left
    # 3 = turn camera right
    # 4 = turn camera up
    # 5 = turn camera down
    # 6 = attack
    # Dataset actions are batched per-key; i.e. dataset_actions['attack'] is an array of length B
    for i in range(batch_size):
        # # TODO this will make all actions invalid. Replace with something
        # # more clever
        # actions[i] = -1
        # raise NotImplementedError("TODO map dataset action at index i to an agent action, or if no mapping, -1")
        # TODO(Jun): This is a very poor lossy mapping; each dataset_action actually can be multi-action,
        # but here we discard everything but the first match (there will be strong bias to 'forward')
        if dataset_actions['forward'][i] == 1:
            actions[i] = 0
        elif dataset_actions['jump'][i] == 1:
            actions[i] = 1
        elif dataset_actions['camera'][i][0] < 0: # TODO(Jun): Make sure the camera works the way I assumed
            actions[i] = 2
        elif dataset_actions['camera'][i][0] > 0:
            actions[i] = 3
        elif dataset_actions['camera'][i][1] > 0:
            actions[i] = 4
        elif dataset_actions['camera'][i][1] < 0:
            actions[i] = 5
        elif dataset_actions['attack'][i] == 1:
            actions[i] = 6
        else:
            actions[i] = -1
    return actions


def train():
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory)
    DATA_DIR = os.getenv("MINERL_DATA_ROOT")
    if DATA_DIR is None:
        DATA_DIR = "."
    # How many times we train over dataset and how large batches we use.
    # Larger batch size takes more memory but generally provides stabler learning.
    EPOCHS = 1
    BATCH_SIZE = 32

    # Create data iterators for going over MineRL data using BufferedBatchIterator
    # Ref: https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#sampling-the-dataset-with-buffered-batch-iter
    data = minerl.data.make(ENV_NAME)
    iterator = minerl.data.BufferedBatchIter(data)

    number_of_actions = 7 # See environment_action_batch_to_agent_actions()
    # # TODO we need to tell the network how many possible actions there are,
    # #      so assign the value in above variable
    # raise NotImplementedError("TODO add number of actions to `number_of_actions`")
    network = NatureCNN((3, 64, 64), number_of_actions).cuda()
    # # TODO create optimizer and loss functions for training
    # #      see examples here https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    # raise NotImplementedError("TODO Create an optimizer and a loss function.")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    loss_function = nn.CrossEntropyLoss()
    # JUN: CONTINUE FROM HERE

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, reward, next_state, done in tqdm(iterator.buffered_batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE)):
        # We only use camera observations here
        obs = dataset_obs["pov"].astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations, otherwise the neural network will get spooked
        obs /= 255.0

        # Turn dataset actions into agent actions
        actions = environment_action_batch_to_agent_actions(dataset_actions)
        assert actions.shape == (obs.shape[0],), "Array from environment_action_batch_to_agent_actions should be of shape {}".format((obs.shape[0],))

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        # TODO perform optimization step:
        # - Predict actions using the neural network (input is `obs`)
        # - Compute loss with the predictions and true actions. Store loss into variable `loss`
        # - Use optimizer to do a single update step
        # See https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html 
        # for a tutorial
        # NOTE: Variables `obs` and `actions` are numpy arrays. You need to convert them into torch tensors.
        obs = torch.tensor(obs)
        assert obs.shape[1:] == (3, 64, 64), f"Expected {(3, 64, 64)}, got {obs.shape[1:]}"
        pred_actions = network(obs.cuda()).cpu()
        actions = torch.tensor(actions)
        assert actions.shape[0] == obs.shape[0], f"Expected {obs.shape[0]}, got {actions.shape[0]}"
        assert len(actions.shape) == 1, f"Expected 1D actions, got {actions.shape}"
        loss = loss_function(pred_actions, actions)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep track of how training is going by printing out the loss
        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    # Store the network
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network, str(MODEL_PATH))
    print("Saved model to", MODEL_PATH)


def enjoy():
    # Load up the trained network
    network = torch.load(str(MODEL_PATH)).cuda()

    env = gym.make(ENV_NAME)

    # Play 10 games with the model
    for game_i in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        reward_sum = 0
        step_count = 0
        with tqdm(total=env.spec.max_episode_steps) as pbar:
            while not done:
                # TODO Process the observation:
                #   - Take only the camera observation
                #   - Add/remove batch dimensions
                #   - Transpose image (needs to be channels-last)
                #   - Normalize image
                #   - Store network output to `logits`
                # For hints, see what preprocessing was done during training
                # raise NotImplementedError("TODO process the observation and run it through network")

                # We only use camera observations here
                obs = obs["pov"].astype(np.float32)
                assert obs.shape == (64, 64, 3), f"Expected {(64, 64, 3)}, got {obs.shape}"
                # Add batch dimension
                obs = np.expand_dims(obs, axis=0)
                # Transpose observations to be channel-first (BCHW instead of BHWC)
                obs = obs.transpose(0, 3, 1, 2)
                # Normalize observations, otherwise the neural network will get spooked
                obs /= 255.0
                assert obs.shape == (1, 3, 64, 64), f"Expected {(1, 3, 64, 64)}, got {obs.shape}"

                logits = network(torch.tensor(obs).cuda())
                # Turn logits into probabilities
                probabilities = torch.softmax(logits, dim=1)[0]
                # Into numpy
                probabilities = probabilities.detach().cpu().numpy()
                # TODO Pick an action based from the probabilities above.
                # The `probabilities` vector tells the probability of choosing one of the agent actions.
                # You have two options:
                # 1) Pick action with the highest probability
                # 2) Sample action based on probabilities
                # Option 2 works better emperically.
                agent_action = np.random.choice(range(len(probabilities)), p=probabilities)

                noop_action = env.action_space.noop()
                environment_action = agent_action_to_environment(noop_action, agent_action)

                obs, reward, done, info = env.step(environment_action)
                reward_sum += reward

                # Show the game situation for us to see what is going on
                # (Note: Normally you would use `environment.render()`, but because of MineRL
                # we have a different setup)
                if step_count % RENDER_EVERY == 0:
                    render(obs, env, headless=HEADLESS, write_frame=True)
                else:
                    render(obs, env, headless=HEADLESS)
                # Wait a moment to give slow brains some time to process the information
                time.sleep(FRAME_DURATION)
                pbar.update(1)
                step_count += 1

                if step_count > OVERRIDE_MAX_STEPS:
                    done = True

        print("Game {}, total reward {}".format(game_i, reward_sum))

        # Save video
        height, width, layers = video_frames[0].shape
        vid_path = MODEL_PATH.parent / Path(MODEL_PATH.stem + f"_{game_i}").with_suffix(".mp4")
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(str(vid_path), fourcc, int(1 / FRAME_DURATION), (width,height))
        for frame in video_frames:
            video.write(frame)
        video.release()
        print("Saved video to", vid_path)
        video_frames.clear()

    env.close()


LOG_PATH = Path("output")
ENV_NAME = 'MineRLTreechop-v0'
MODEL_PATH = LOG_PATH / ENV_NAME / "behavioural_cloning.pth"
FRAME_DURATION = 0.05
RENDER_EVERY = 1
OVERRIDE_MAX_STEPS = 200
NUM_EPISODES = 3
HEADLESS = True
video_frames = []

if __name__ == "__main__":
    envs = [
        "MineRLTreechop-v0",
        "MineRLBasaltBuildVillageHouse-v0", 
        "MineRLBasaltCreatePlainsAnimalPen-v0", 
        "MineRLBasaltCreateVillageAnimalPen-v0", 
        "MineRLBasaltFindCave-v0", 
        "MineRLBasaltMakeWaterfall-v0"
    ]
    for env in envs:
        print(f"\n\n\nNow working on {env}")
        ENV_NAME = env
        MODEL_PATH = LOG_PATH / ENV_NAME / "behavioural_cloning.pth"
        # First train the model...
        train()
        # ... then play it on the environment to see how it does
        enjoy()
