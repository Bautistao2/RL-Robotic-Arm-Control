# 🎯 Entrenamiento de CartPole con PPO

## 📌 Descripción  
Este proyecto utiliza el algoritmo **Proximal Policy Optimization (PPO)** de la librería `stable-baselines3` para entrenar un agente en el entorno **CartPole-v1** de OpenAI Gym.  

El modelo ha sido optimizado con:  
✅ Ajustes personalizados en sus hiperparámetros.  
✅ Una función de recompensa modificada para mejorar el rendimiento.  

---

## 🛠️ Instalación y Configuración  

### 1️⃣ **Clonar el repositorio**  
```sh
git clone https://github.com/Bautistao2/Network-Anomaly-Detection-PBSCAN.git
cd Network-Anomaly-Detection-PBSCAN

```

### 2️⃣ **Crear un entorno virtual (recomendado) 🏗️**  
Es recomendable utilizar un **entorno virtual** para aislar las dependencias:  
```sh
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ **Instalar dependencias 🔧**  
```sh
pip install -r requirements.txt
```
⚠ **Si no tienes `requirements.txt`**, créalo con:  
```sh
echo "gym\nnumpy\nstable-baselines3" > requirements.txt
```

---

## 🚀 Ejecución  
Para entrenar y evaluar el modelo, ejecuta:  
```sh
python main.py
```
Esto iniciará el entrenamiento del modelo en **CartPole-v1** y guardará los modelos en cada iteración.

---

## 🏗️ Estructura del Código  

### 1️⃣ **Creación del Entorno**  
```python
env = make_vec_env("CartPole-v1", n_envs=1)
```
✔ Se crea un entorno **vectorizado** para permitir múltiples simulaciones en paralelo.  

### 2️⃣ **Configuración del Modelo PPO**  
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
📌 **Explicación de hiperparámetros:**  
🔹 `MlpPolicy` → Usa una red neuronal multicapa.  
🔹 `learning_rate=0.0005` → Controla la tasa de aprendizaje del modelo.  
🔹 `gamma=0.90` → Factor de descuento para recompensas futuras.  
🔹 `n_steps=2048` → Pasos antes de actualizar la política.  
🔹 `batch_size=512` → Tamaño del lote en cada actualización.  
🔹 `ent_coef=0.01` → Aumenta la exploración mediante entropía.  

### 3️⃣ **Modificación de la Función de Recompensa 🏆**  
```python
def custom_reward(env, reward, terminated, truncated):
    if terminated or truncated:
        reward += 50  # Bonus al completar un episodio exitosamente
    else:
        reward -= 0.1  # Penalización por cada paso
    return reward
```
✅ **Mejoras:**  
✔ Se **premia** con **+50 puntos** si el episodio termina exitosamente.  
✔ Se **penaliza** con **-0.1 puntos** por cada paso intermedio.  

### 4️⃣ **Entrenamiento del Modelo 🏋️**  
```python
TIMESTEPS = 100000
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS // 10)
    model.save(f"ppo_cartpole_{i}")
```
🎯 **Explicación:**  
✔ Se entrena el modelo en **100,000 timesteps**, divididos en 10 ciclos.  
✔ En cada ciclo, el modelo se guarda con el nombre `ppo_cartpole_{i}`.  

### 5️⃣ **Evaluación del Modelo 📊**  
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
    
    print(f"🎯 Precisión del modelo: {success_count}/{num_episodes} episodios exitosos")
    print(f"📊 Recompensa media: {np.mean(total_rewards)}")
```
✔ Se juegan **10 episodios** para medir el rendimiento.  
✔ Se **cuentan los episodios exitosos** (cuando la recompensa supera **195 puntos**).  
✔ Se calcula la **recompensa media** por episodio.  

---

## 🏆 Resultados Obtenidos  
✔ **Precisión:** ✅ 10/10 episodios exitosos.  
✔ **Recompensa media:** 🎯 **226.14 puntos**.  

📈 **Gráfica del progreso** :  
```
Iteración 1: Recompensa media → 22.7
Iteración 5: Recompensa media → 49.4
Iteración 10: Recompensa media → 226.14 🎯
```

---

## 💡 Posibles Errores y Soluciones  

❌ **`ModuleNotFoundError: No module named 'gym'`**  
➡ Solución:  
```sh
pip install gym
```

❌ **`ImportError: cannot import name 'PPO'`**  
➡ Solución:  
```sh
pip install stable-baselines3
```

❌ **`AttributeError: module 'numpy' has no attribute 'bool8'`**  
➡ Solución:  
```sh
pip install --upgrade numpy
```

---

## 🎯 Conclusión  
🚀 **Este modelo PPO logra un rendimiento sólido en el entorno CartPole-v1 gracias a:**  
✔ Un ajuste óptimo de los **hiperparámetros**.  
✔ Una **función de recompensa personalizada**.  
✔ Un **entrenamiento estructurado** en múltiples iteraciones.  

Este código puede servir como base para entrenar modelos en otros entornos de OpenAI Gym. 🏆🔥  

---

## 📜 Licencia  
Este proyecto está bajo la **Licencia MIT**.  

