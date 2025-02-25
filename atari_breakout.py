import gymnasium as gym
import ale_py
import numpy as np
import cv2
import time

# Global variable to store last known paddle x-position
last_paddle_x = None

def process_frame(gray):
    """
    Process the grayscale frame to detect the ball and paddle.
    We crop out the score area and side borders to avoid detection issues.
    """
    global last_paddle_x
    # Crop to the game area: remove top score area and side borders.
    # For example, use rows 34:194 and columns 15:147.
    cropped = gray[34:194, 15:147]  # This yields a smaller, border-free view.

    # Use a lower threshold value to capture bright objects
    _, thresh = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours for debugging
    contour_img = cropped.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    cv2.imshow("Contours", contour_img)
    cv2.waitKey(1)
    
    ball_pos = None
    paddle_pos = None
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # Uncomment the next line for per-contour debugging:
        # print(f"Contour: x={x}, y={y}, w={w}, h={h}, area={area}")
        
        # Detect the ball:
        # Assume the ball is a very small, nearly square blob in the upper region.
        if 1 < area < 10 and abs(w - h) < 3 and y < 100:
            ball_pos = (x + w // 2, y + h // 2)
        
        # Detect the paddle:
        # Assume the paddle is a wide, horizontal object near the bottom.
        if y > 110 and w > 15 and w > h * 3:
            paddle_pos = (x + w // 2, y + h // 2)
            last_paddle_x = paddle_pos[0]  # update the last known paddle x
    
    # If paddle detection fails, try using the last known position.
    if paddle_pos is None and last_paddle_x is not None:
        # Assume the paddle is at the last known x with a fixed y (bottom region)
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
    
    return ball_pos, paddle_pos

def simple_breakout_ai(observation):
    """
    Heuristic: move the paddle horizontally toward the ball's x position.
    Action mapping for Breakout (ALE):
       0 : NOOP
       1 : FIRE
       2 : Move right
       3 : Move left
    """
    ball_pos, paddle_pos = process_frame(observation)
    action = 0  # Default is NOOP
    
    if ball_pos and paddle_pos:
        ball_x = ball_pos[0]
        paddle_x = paddle_pos[0]
        # Increase the margin (hysteresis) to reduce rapid switching.
        if paddle_x < ball_x - 10:
            action = 2  # Move right
        elif paddle_x > ball_x + 10:
            action = 3  # Move left
        
        print(f" | Heuristic Action: {action}", end="\r")
    else:
        print(" | Default Action: NOOP", end="\r")
    
    return action

if __name__ == "__main__":
    # Register ALE environments
    gym.register_envs(ale_py)
    
    # Create the Breakout environment with grayscale observations and human rendering.
    env = gym.make("ALE/Breakout-v5", render_mode="human", obs_type="grayscale")
    
    obs, _ = env.reset()
    print("Initial observation shape:", obs.shape)
    
    episode_steps = 0
    total_reward = 0
    ball_fired = False  # Indicates whether a ball is in play
    done = False
    
    while not done:
        if not ball_fired:
            # Automatically fire until the ball is launched.
            action = 1  # FIRE action
            print("Firing ball...", end="\r")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            
            # Check if the ball appears after firing.
            ball_pos, _ = process_frame(obs)
            if ball_pos is not None:
                ball_fired = True
                print("\nBall launched, switching to heuristic control.")
        else:
            # Check if the ball is still in play.
            ball_pos, _ = process_frame(obs)
            if ball_pos is None:
                print("\nBall lost, waiting 1 sec to fire again.")
                time.sleep(1)
                ball_fired = False
                continue  # Skip stepping the environment this cycle.
            
            # Use heuristic control once the ball is in play.
            action = simple_breakout_ai(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            print(f"Reward: {reward} | Info: {info}", end="\r")
        
        # Add a short delay to reduce processing frequency and smooth motion.
        time.sleep(0.05)
        
        if terminated or truncated:
            print(f"\nEpisode finished after {episode_steps} steps. Total reward: {total_reward}")
            done = True
    
    env.close()
    cv2.destroyAllWindows()

