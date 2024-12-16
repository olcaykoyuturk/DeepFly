import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

DATE_FORMAT = "%m-%d %H:%M:%S"

# Çalışma bilgilerini kaydetmek için dizin
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

# Kullanılacak cihazı belirle (GPU varsa kullanılır, yoksa CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set, hyperparameter_file="hyperparameters.yml"):
        with open(hyperparameter_file, 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag

        # Neural Network
        self.loss_fn = nn.MSELoss()          # Mean Squared Error
        self.optimizer = None                # Optimizasyon algoritması, daha sonra tanımlanacak

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

       # Ortamı oluştur (Çevre parametrelerini hiperparametrelerden alarak oluşturulur)
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Mümkün olan eylem sayısını al
        num_actions = env.action_space.n

        # Gözlem alanı boyutunu al
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # Bölüm başına ödülleri takip etmek için liste
        rewards_per_episode = []

        # Policy ve hedef ağları oluştur
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)

        if is_training:
            # Epsilon'u başlat
            epsilon = self.epsilon_init

            # Replay memory'i başlat
            memory = ReplayMemory(self.replay_memory_size)

            # Hedef ağı oluştur ve policy ağına eşitle
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy ağı optimizasyon aracı (Adam optimizer)
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # Epsilon azalma geçmişini takip et
            epsilon_history = []

            # Adım sayacı (policy => hedef ağı senkronizasyonu için kullanılır)
            step_count=0

            # En iyi ödül izleyici
            best_reward = -9999999
        else:
            # Öğrenilmiş policy'yi yükle
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # Modeli değerlendirme moduna al
            policy_dqn.eval()

        # Eğitimi sınırsız olarak devam ettir (kullanıcı manuel olarak durdurabilir)
        for episode in itertools.count():

            state, _ = env.reset()  # Ortamı sıfırla (dönüş: (state, info))
            state = torch.tensor(state, dtype=torch.float, device=device) # Durumu tensöre çevir ve cihazda işle

            terminated = False      # Ajan hedefe ulaştığında veya başarısız olduğunda True olur
            episode_reward = 0.0    # Bölüm başına ödülleri toplamak için

            # Bölüm sona erene kadar veya maksimum ödüle ulaşana kadar eylem gerçekleştir
            while(not terminated and episode_reward < self.stop_on_reward):

                # Epsilon-greedy stratejisine göre eylem seçimi
                if is_training and random.random() < epsilon:
                    # Rastgele eylem seç
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # En iyi eylemi seç
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch, bir batch katmanı bekler, bu yüzden boyut eklenir.
                        # policy_dqn tensor([[1], [2], [3]]) döner, bu yüzden squeeze ile tek boyuta indirilir.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Eylemi gerçekleştir (truncated ve info kullanılmaz)
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Ödülleri biriktir
                episode_reward += reward

                # Yeni durumu ve ödülü tensöre çevir
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Deneyimi belleğe kaydet
                    memory.append((state, action, new_state, reward, terminated))

                    # Adım sayısını artır
                    step_count+=1

                # Bir sonraki duruma geç
                state = new_state

            # Bölüm başına ödülleri takip et
            rewards_per_episode.append(episode_reward)

            # Yeni en iyi ödül elde edildiğinde modeli kaydet
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Grafiği her x saniyede bir güncelle
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # Yeterli deneyim toplandığında
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Epsilon'u azalt
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Belirli bir adım sayısından sonra policy ağını hedef ağına kopyala
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0


    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Ortalama ödülleri (Y ekseni) ve bölümleri (X ekseni) çiz
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # 1 satır x 2 sütunlu ızgarada 1. hücreye çiz
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # 1 satır x 2 sütunlu ızgarada 2. hücreye çiz
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Grafikleri kaydet
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Politika ağını optimize et
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Deneyim listesini ayır ve her bir elemanı ayrı ayrı al
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN için en iyi eylemleri policy ağından al
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        #  Hedef Q değerlerini (beklenen getiriler) hesapla
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Kayıp fonksiyonunu hesapla
        loss = self.loss_fn(current_q, target_q)

        # Modeli optimize et (geri yayılım)
        self.optimizer.zero_grad()  # Gradientleri temizle
        loss.backward()             # Gradientleri hesapla
        self.optimizer.step()       # Ağın parametrelerini güncelle (örneğin ağırlıklar ve biaslar)

if __name__ == '__main__':
    # Komut satırı girdilerini işle
    parser = argparse.ArgumentParser(description='Modeli eğit veya test et.')
    parser.add_argument('hyperparameters', help='Hiperparametre seti')
    parser.add_argument('--train', help='Eğitim modu', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)