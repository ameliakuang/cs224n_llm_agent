import os
import ale_py
import cv2
import logging
import datetime
import sys
from pathlib import Path
import numpy as np
import io
import contextlib

from dotenv import load_dotenv
from autogen import config_list_from_json
import gymnasium as gym
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError

load_dotenv()
gym.register_envs(ale_py)

# Global variable to store last known paddle x-position
last_paddle_x = None

def process_image(obs):
    """
    Process the grayscale image to detect the ball and paddle in Breakout.
    
    Args:
        obs: Grayscale image of the game screen.
        
    Returns:
        dict: Dictionary containing the positions of the ball and paddle in the form:
              {"ball_pos": (x, y) or None, "paddle_pos": (x, y) or None, "reward": np.nan}
    """
    global last_paddle_x
    # Crop to the game area (removing top score area and side borders)
    cropped = obs[34:194, 15:147]
    
    # Use a lower threshold value to capture bright objects (ball and paddle)
    _, thresh = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)
    
    # (Optional) Visualization - commented out for optimization runs.
    # cv2.imshow("Threshold", thresh)
    # cv2.waitKey(1)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # (Optional) Draw contours for debugging
    # contour_img = cropped.copy()
    # cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("Contours", contour_img)
    # cv2.waitKey(1)
    
    ball_pos = None
    paddle_pos = None
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Detect the ball: a small, roughly square blob in the upper region.
        if 1 < area < 10 and abs(w - h) < 3 and y < 100:
            ball_pos = (x + w // 2, y + h // 2)
        
        # Detect the paddle: a wide, horizontal object near the bottom.
        if y > 110 and w > 15 and w > h * 3:
            paddle_pos = (x + w // 2, y + h // 2)
            last_paddle_x = paddle_pos[0]
    
    # If paddle detection fails, use the last known paddle x-position.
    if paddle_pos is None and last_paddle_x is not None:
        paddle_pos = (last_paddle_x, 120)
        print("Using last known paddle position.", end=" | ")
    
    if ball_pos:
        print(f"Detected ball at: {ball_pos}", end=" | ")
    else:
        print("Ball not detected", end=" | ")
        
    if paddle_pos:
        print(f"Detected paddle at: {paddle_pos}", end=" | ")
    else:
        print("Paddle not detected", end=" | ")
    
    return {"ball_pos": ball_pos, "paddle_pos": paddle_pos, "reward": np.nan}

class BreakoutTracedEnv:
    def __init__(self, 
                 env_name="ALE/Breakout-v5",
                 render_mode="human",
                 obs_type="grayscale"):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.env = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = gym.make(self.env_name, render_mode=self.render_mode, obs_type=self.obs_type)
        self.env.reset()
        self.obs = None
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()
    
    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        obs, info = self.env.reset()
        self.obs = process_image(obs)
        self.obs['reward'] = np.nan
        return self.obs, info
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = next_obs = process_image(next_obs)
            self.obs['reward'] = next_obs['reward'] = reward
        except Exception as e:
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The operator step raises an exception.",
                name="exception_step",
            )
            raise ExecutionError(e_node)
        
        @bundle()
        def step_bundle(action):
            """
            Take action in the environment and return the next observation.
            """
            return next_obs

        next_obs = step_bundle(action)
        return next_obs, reward, termination, truncation, info

def rollout(env, horizon, policy):
    """Rollout a policy in an environment for a given horizon."""
    try:
        obs, _ = env.reset()
        trajectory = dict(observations=[], actions=[], rewards=[], terminations=[], truncations=[], infos=[], steps=0)
        trajectory["observations"].append(obs)
        
        for _ in range(horizon):
            error = None
            try:
                action = policy(obs)
                next_obs, reward, termination, truncation, info = env.step(action)
            except trace.ExecutionError as e:
                error = e
                reward = np.nan
                termination = True
                truncation = False
                info = {}
            
            if error is None:
                trajectory["observations"].append(next_obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminations"].append(termination)
                trajectory["truncations"].append(truncation)
                trajectory["infos"].append(info)
                trajectory["steps"] += 1
                if termination or truncation:
                    break
                obs = next_obs
    finally:
        env.close()
    
    return trajectory, error

def optimize_policy(
    env_name="ALE/Breakout-v5",
    horizon=800,
    memory_size=5,
    n_optimization_steps=5,
    verbose=False,
    model="gpt-4o-mini"
):
    @trace.bundle(trainable=True)
    def policy(obs):
        '''
        A policy that moves the paddle horizontally toward the ball.
        If the ball is not detected, return FIRE to launch the ball.
        
        Action mapping for Breakout:
            0 : NOOP
            1 : FIRE
            2 : Move right
            3 : Move left
        
        Args:
            obs (dict): A dictionary with keys "ball_pos" and "paddle_pos" representing 
                        the positions of the ball and paddle.
        Output:
            action (int): The action to take.
        '''
        ball_pos = obs["ball_pos"]
        paddle_pos = obs["paddle_pos"]

        # If the ball is not detected, attempt to launch it.
        if ball_pos is None:
            return 1  # FIRE
        
        action = 0  # Default NOOP
        if ball_pos and paddle_pos:
            ball_x = ball_pos[0]
            paddle_x = paddle_pos[0]
            if paddle_x < ball_x - 10:
                action = 2  # Move right
            elif paddle_x > ball_x + 10:
                action = 3  # Move left
        return action
    
    # Retrieve configuration list for the optimizer.
    config_path = os.getenv("OAI_CONFIG_LIST")
    config_list = config_list_from_json(config_path)
    config_list = [config for config in config_list if config["model"] == model]
    optimizer = OptoPrime(policy.parameters(), config_list=config_list, memory_size=memory_size)
    
    env = BreakoutTracedEnv(env_name=env_name)
    logger.info("Optimization Starts")
    rewards = []
    try:
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            if error is None:
                total_reward = sum(traj['rewards'])
                feedback = f"Episode ends after {traj['steps']} steps with total score: {total_reward:.1f}"
                if total_reward > 0:
                    feedback += "\nGood job! You've hit some bricks."
                else:
                    feedback += "\nTry to improve paddle positioning to hit the ball and break bricks."
                target = traj['observations'][-1]
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}, Parameter: {policy.parameters()}")

            instruction = (
                "In Breakout, you control a paddle at the bottom of the screen to bounce a ball "
                "and break bricks. Fire the ball to launch it and then move the paddle left or right "
                "to intercept the ball. The goal is to score points by breaking bricks while keeping "
                "the ball in play. "
            )
            optimizer.objective = instruction + optimizer.default_objective
            
            optimizer.zero_feedback()
            optimizer.backward(target, feedback, visualize=True)
            logger.info(optimizer.problem_instance(optimizer.summarize()))
            
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                optimizer.step(verbose=verbose)
                llm_output = stdout_buffer.getvalue()
                if llm_output:
                    logger.info(f"LLM response:\n{llm_output}")
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}, Parameter: {policy.parameters()}")
            rewards.append(total_reward)
    finally:
        if env is not None:
            env.close()
    
    avg_reward = sum(rewards) / len(rewards) if rewards else float('nan')
    logger.info(f"Final Average Reward: {avg_reward}")
    return rewards

if __name__ == "__main__":
    # Set up logging to console and file.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Set up file logging.
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"breakout_ai_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting Breakout AI training...")
    rewards = optimize_policy(
        env_name="ALE/Breakout-v5",
        horizon=800,
        n_optimization_steps=5,
        memory_size=5,
        verbose='output',
        model="gpt-4o-mini"
    )
    logger.info("Training completed.")
