import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Crear el entorno
env = make_vec_env("CartPole-v1", n_envs=1)

# Ajustar hiperparámetros del modelo
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=0.0005,
    gamma=0.90,
    n_steps=2048,
    batch_size=512,
    ent_coef=0.01,
    verbose=1
)

# Modificar la función de recompensa

def custom_reward(reward, terminated, truncated):
    if terminated or truncated:
        return reward + 50  # Aumentar recompensa si el episodio termina
    return reward - 0.1  # Penalización por cada paso

# Entrenar el modelo
TIMESTEPS = 100000
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS // 10)
    model.save(f"ppo_cartpole_{i}")

# Evaluar el modelo
def evaluate(model, num_episodes=10):
    env = gym.make("CartPole-v1")
    success_count = 0
    total_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset() if hasattr(env, 'reset') and callable(env.reset) else (env.reset(), {})
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result  # Gym 26+
            else:
                obs, reward, done, _ = step_result  # Gym <26
                terminated, truncated = bool(done), False  # Convertir a bool por compatibilidad
            
            reward = custom_reward(reward, terminated, truncated)
            total_reward += reward
            done = terminated or truncated
        
        if total_reward > 195:
            success_count += 1
        total_rewards.append(total_reward)
    
    env.close()  # Cerrar entorno al final
    print(f"Precisión del modelo: {success_count}/{num_episodes} episodios exitosos")
    print(f"Recompensa media por episodio: {np.mean(total_rewards)}")

evaluate(model)
