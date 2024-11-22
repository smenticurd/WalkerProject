import numpy as np
import sys
import gym
import torch
import random
from td3 import TD3
from buffer import ExperienceReplay
import matplotlib.pyplot as plt

def main():
    env = gym.make('BipedalWalker-v3')

    # Set seed for reproducible results
    seed = 1
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    buffer_size = 1000000
    batch_size = 100
    noise = 0.1

    # Device configuration
    device = torch.device("cpu")  # Change this to "cuda" if you want to use GPU.

    # Initialize the policy
    policy = TD3(state_dim, action_dim, max_action, env, device)
    
    try:
        print("Loading previous model")
        policy.load()
    except Exception as e:
        print('No previous model to load. Training from scratch.')

    # Experience replay buffer
    buffer = ExperienceReplay(buffer_size, batch_size, device)

    # Training hyperparameters
    save_score = 400
    episodes = 650
    timesteps = 2000

    # Tracking variables
    best_reward = -1*sys.maxsize
    scores_over_episodes = []

    for episode in range(episodes):
        avg_reward = 0
        state = env.reset()
        for i in range(timesteps):
            # Select action with added noise
            action = policy.select_action(state) + np.random.normal(0, max_action * noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)
            
            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)

            # Store the experience in the buffer
            buffer.store_transition(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state
            avg_reward += reward

            # Train the agent after collecting a batch of experiences
            if len(buffer) > batch_size:
                policy.train(buffer, i)

            # Render the environment (if needed)
            env.render()

            if done or i > timesteps:
                scores_over_episodes.append(avg_reward)
                print(f'Episode {episode} finished with reward: {avg_reward}')
                print(f'Finished at timestep {i}')
                break
        
        # Save the model if performance improves
        if np.mean(scores_over_episodes[-50:]) > save_score:
            print(f'Saving agent - past 50 scores gave better avg than {save_score}')
            best_reward = np.mean(scores_over_episodes[-50:])
            save_score = best_reward
            policy.save()
            break  # Saving the model, stopping training early if performance is good.

        # Regular saving if episode score improves
        if episode >= 0 and avg_reward > best_reward:
            print(f'Saving agent - score for this episode was better than the best-known score..')
            best_reward = avg_reward
            policy.save()  # Save current policy + optimizer
    
    # Plotting the scores
    fig = plt.figure()
    plt.plot(np.arange(1, len(scores_over_episodes) + 1), scores_over_episodes)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

main()
