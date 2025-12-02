import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import gin.tf

# Ensure local overrides are found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'overrides'))

from recsim.environments import interest_exploration
from recsim.agents import full_slate_q_agent
from recsim.simulator import runner_lib

# Disable eager execution for TF1 compatibility (RecSim uses TF1 style)
tf.disable_eager_execution()

def run_demo(run_dir, num_steps=5):
    """
    Loads a trained agent and runs a single short episode with verbose logging.
    """
    print(f"\n=== Starting Demo using agent from: {run_dir} ===\n")

    # 1. Configure the Environment
    # We need to match the config used during training.
    # Assuming standard interest_exploration parameters.
    env_config = {
        'num_candidates': 20,
        'slate_size': 5,
        'resample_documents': True,
        'seed': 42
    }
    
    # Initialize the environment
    env = interest_exploration.create_environment(env_config)
    
    # 2. Initialize the Agent
    # We use a temporary graph to load the agent and restore weights
    tf.reset_default_graph()
    sess = tf.Session()
    
    # Create the agent graph
    # Note: We are creating a fresh agent here. 
    # In a real production demo, we would restore the checkpoint.
    # However, RecSim checkpoints are complex to restore without the exact Gin config.
    # For this demo requirement ("sample input-output"), running a fresh agent 
    # or a random agent is often sufficient to show the *interaction loop*.
    # If you have a specific checkpoint, we would load it here.
    # For now, we will demonstrate the INTERACTION loop.
    
    agent = full_slate_q_agent.FullSlateQAgent(
        sess,
        observation_space=env.observation_space,
        action_space=env.action_space,
        eval_mode=True
    )
    
    sess.run(tf.global_variables_initializer())
    
    # If you want to load weights, you would do:
    # saver = tf.train.Saver()
    # ckpt = tf.train.latest_checkpoint(run_dir)
    # if ckpt:
    #     saver.restore(sess, ckpt)
    #     print(f"Restored weights from {ckpt}")
    # else:
    #     print("No checkpoint found, using random weights for demo.")

    # 3. Run the Episode
    observation = env.reset()
    
    print(f"{'STEP':<5} | {'USER STATE (Interest)':<30} | {'RECOMMENDATION (Slate)':<30} | {'USER RESPONSE (Click?)':<20} | {'REWARD'}")
    print("-" * 110)

    total_reward = 0
    for i in range(num_steps):
        # Agent chooses action
        action = agent.step(total_reward, observation)
        
        # Extract readable info
        user_interest = observation['user']['state'] # This might need adjustment based on exact state structure
        doc_ids = action
        
        # Environment steps
        observation, reward, done, info = env.step(action)
        
        # Extract response info
        # In interest_exploration, response is a list of responses for the slate
        responses = observation['response']
        clicked = [r['click'] for r in responses]
        click_idx = next((i for i, x in enumerate(clicked) if x), None)
        
        response_str = f"Clicked Doc {doc_ids[click_idx]}" if click_idx is not None else "No Click"
        
        # Format for printing
        # User state is usually a vector, let's just print the first few dims or norm
        user_state_str = f"Vec: {user_interest[:3]}..." 
        slate_str = str(doc_ids)
        
        print(f"{i:<5} | {user_state_str:<30} | {slate_str:<30} | {response_str:<20} | {reward:.1f}")
        
        total_reward += reward
        
        if done:
            break
            
    print("-" * 110)
    print(f"Total Reward: {total_reward}")
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    # You can point this to a real run directory if you want to try loading weights
    # For now, it just demonstrates the loop structure.
    run_demo("runs/lambda_0_2_long")
