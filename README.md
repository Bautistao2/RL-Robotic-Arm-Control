# 🎯 Entrenamiento de CartPole con PPO

## 📌 Descripción
Este proyecto utiliza el algoritmo **Proximal Policy Optimization (PPO)** de la librería `stable-baselines3` para entrenar un agente en el entorno **CartPole-v1** de OpenAI Gym. El modelo ha sido optimizado con ajustes personalizados en sus hiperparámetros y una función de recompensa modificada.

## 🛠️ Requisitos
Antes de ejecutar el script, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install gym numpy stable-baselines3
```

## 🚀 Uso
Para entrenar y evaluar el modelo, ejecuta el siguiente comando:

```bash
python robotic3.py
```

## 🏗️ Estructura del Código

### 1️⃣ Creación del Entorno
```python
env = make_vec_env("CartPole-v1", n_envs=1)
```
Se crea un entorno vectorizado para facilitar el entrenamiento en paralelo. En este caso, usamos una sola instancia del entorno.

### 2️⃣ Configuración del Modelo PPO
```python
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
```
Se instancia un modelo PPO con:
- **MlpPolicy**: Red neuronal multicapa para tomar decisiones.
- **learning_rate** = 0.0005: Tasa de aprendizaje.
- **gamma** = 0.90: Factor de descuento para futuras recompensas.
- **n_steps** = 2048: Número de pasos de simulación antes de actualizar la política.
- **batch_size** = 512: Tamaño del lote para entrenar la red neuronal.
- **ent_coef** = 0.01: Coeficiente para la entropía, que fomenta la exploración.

### 3️⃣ Modificación de la Función de Recompensa 🏆
```python
def custom_reward(env, reward, terminated, truncated):
    if terminated or truncated:
        reward += 50  # Recompensa adicional al completar un episodio
    else:
        reward -= 0.1  # Penalización por cada paso intermedio
    return reward
```
Se redefine la recompensa:
- Se **añaden 50 puntos** si el episodio termina exitosamente.
- Se **penaliza con -0.1 puntos** cada paso para fomentar estrategias eficientes.

### 4️⃣ Entrenamiento del Modelo 🏋️
```python
TIMESTEPS = 100000
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS // 10)
    model.save(f"ppo_cartpole_{i}")
```
Se entrena el modelo en **100,000 timesteps**, dividiendo el proceso en 10 iteraciones y guardando el modelo en cada una de ellas.

### 5️⃣ Evaluación del Modelo 📊
```python
def evaluate(model, num_episodes=10):
    env = gym.make("CartPole-v1")
    success_count = 0
    total_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
            else:
                obs, reward, done, _ = step_result
                terminated, truncated = done, False
            
            reward = custom_reward(env, reward, terminated, truncated)
            total_reward += reward
            done = terminated or truncated
        
        if total_reward > 195:
            success_count += 1
        total_rewards.append(total_reward)
    
    print(f"Precisión del modelo: {success_count}/{num_episodes} episodios exitosos")
    print(f"Recompensa media por episodio: {np.mean(total_rewards)}")
```
La evaluación:
- Juega **10 episodios** y mide el rendimiento del modelo.
- Cuenta episodios exitosos (cuando la recompensa supera **195 puntos**).
- Calcula la **recompensa media** por episodio.

### 🏆 Resultados Obtenidos
Tras el entrenamiento, el modelo logró:
✅ **Precisión**: 10/10 episodios exitosos.  
✅ **Recompensa media**: 226.14 puntos.  

## 🎯 Conclusión
Este modelo PPO logra un rendimiento sólido en el entorno CartPole-v1 gracias a:
- Un ajuste fino de hiperparámetros.
- Una función de recompensa mejorada.
- Un entrenamiento estructurado en múltiples iteraciones.

Este código puede servir como base para entrenar modelos más avanzados en otros entornos de OpenAI Gym. 🚀

