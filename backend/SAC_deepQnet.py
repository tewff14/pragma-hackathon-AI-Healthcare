import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt

device = torch.device("cpu")
# --- 1. ARSITEKTUR AUTOENCODER (Improved for MIMIC) ---
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=128, latent_dim=24):
        super(AutoEncoder, self).__init__()
        # Encoder: 37 -> 64 -> 24 (Compression)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Normalisasi Latent ke range [-1, 1]
        )
        # Decoder: 24 -> 64 -> 37 (Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# --- 2. FUNGSI TRAINING KHUSUS (Menerima Xtrain & Xvalidat Numpy) ---
def train_autoencoder_mimic(X_train_np, X_val_np, input_dim=37, epochs=50, batch_size=256, lr=1e-3, save_path='best_autoencoder_mimic.pth'):
    
    # Konversi Numpy ke Tensor
    train_tensor = torch.FloatTensor(X_train_np)
    val_tensor = torch.FloatTensor(X_val_np)
    
    # Buat DataLoader
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n[AutoEncoder] Start Training on {len(X_train_np)} samples...")

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler & Logging
        scheduler.step(avg_val_loss)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"[AutoEncoder] Early stopping at epoch {epoch+1}")
                break
    
    print(f"[AutoEncoder] Training Finished. Best Model Saved to {save_path}")
    return model, history

def plot_reconstruction(model, X_data, index=0):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        original = torch.FloatTensor(X_data[index]).to(device).unsqueeze(0)
        reconstructed = model(original)
    
    orig = original.cpu().numpy().flatten()
    recon = reconstructed.cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 4))
    plt.plot(orig, label='Original (Normalized)', marker='o', alpha=0.7)
    plt.plot(recon, label='Reconstructed', linestyle='--', linewidth=2)
    plt.title(f"Reconstruction Check (Sample {index})")
    plt.legend()
    plt.show()


class Actor(nn.Module):
    # Tambahkan action_min dan action_max di init
    def __init__(self, state_dim, action_dim, hidden_dim=512, action_min=None, action_max=None):
        super(Actor, self).__init__()
        
        # Golden Tuning: Wide Network + GELU
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # --- LOGIKA ACTION RESCALING ---
        if action_min is not None and action_max is not None:
            # Pastikan format tensor float32 dan masuk ke buffer model
            # Rumus: Scale = (Max - Min) / 2
            # Rumus: Bias  = (Max + Min) / 2
            self.register_buffer('action_scale', torch.tensor((action_max - action_min) / 2.0, dtype=torch.float32))
            self.register_buffer('action_bias', torch.tensor((action_max + action_min) / 2.0, dtype=torch.float32))
            print(f"[Actor] Rescaling enabled: Scale={self.action_scale}, Bias={self.action_bias}")
        else:
            # Default ke rentang -1 s.d 1 jika tidak ada info
            self.register_buffer('action_scale', torch.ones(action_dim))
            self.register_buffer('action_bias', torch.zeros(action_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2) 
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() 
        
        # 1. Output Tanh (Selalu -1 s.d 1)
        y_t = torch.tanh(x_t)
        
        # 2. Rescaling ke Range Data Asli (PENTING!)
        # Action = tanh * scale + bias
        action = y_t * self.action_scale + self.action_bias

        # 3. Hitung Log Prob dengan koreksi Scaling
        log_prob = normal.log_prob(x_t)
        
        # Koreksi 1: Tanh transformation
        log_prob -= 2 * (np.log(2) - x_t - F.softplus(-2 * x_t))
        
        # Koreksi 2: Linear Scaling transformation
        # Karena kita mengalikan aksi dengan 'scale', densitas probabilitas berubah
        # log(p(y)) = log(p(x)) - log(scale)
        log_prob -= torch.log(self.action_scale)
        
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Critic, self).__init__()
        
        # 1. ARSITEKTUR: Ganti ReLU dengan Mish
        # Q-Function surface itu sangat rumit. Mish membantu memperhalus landscape loss
        # sehingga gradien mengalir lebih baik ke layer awal.
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 2. INISIALISASI: Orthogonal Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Gain 1.0 biasanya cukup untuk Critic, atau sqrt(2) jika ingin agresif
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        # Pastikan input dim sesuai sebelum concat
        x = torch.cat([state,  action], dim=1)
        return self.net(x)

class SACAgent:
    # Tambahkan argumen action_space_min dan action_space_max
    def __init__(self, state_dim=24, action_dim=2, gamma=0.99, tau=0.005, 
                 alpha=0.2, bc_weight=0.1, safety_weight=1.0,
                 action_space_min=None, action_space_max=None): # <--- PARAMETER BARU
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pass min/max ke Actor saat inisialisasi
        self.actor = Actor(state_dim, action_dim, hidden_dim=256, 
                           action_min=action_space_min, 
                           action_max=action_space_max).to(self.device)
                           
        # Critic tidak butuh rescaling karena dia menerima (state, action) apa adanya
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        # ... (Sisa kode __init__ sama persis: optimizer, scheduler, dll) ...
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=1e-4) # Ingat LR Actor lebih kecil
        self.critic_1_optimizer = optim.AdamW(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = optim.AdamW(self.critic_2.parameters(), lr=3e-4)
        
        # ... (Copy sisa atribut self.gamma, self.autoencoder, self.norm_stats, dll) ...
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.5, patience=5)
        self.critic_1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.critic_1_optimizer, mode='min', factor=0.5, patience=5)
        self.critic_2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.critic_2_optimizer, mode='min', factor=0.5, patience=5)
        
        self.autoencoder = None
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.safety_weight = safety_weight
        
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=3e-4)
        self.alpha_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.alpha_optimizer, mode='min', factor=0.5, patience=5)
        
        self.norm_stats = self._load_stats()
    def _load_stats(self):
        """Helper internal untuk memuat statistik normalisasi"""
        try:
            with open('app/data/state_norm_stats.pkl', 'rb') as f:
                stats = pickle.load(f)
            print("✓ [SACAgent] State normalization stats loaded.")
            return stats
        except Exception as e:
            print(f"! [SACAgent] WARNING: Gagal memuat 'state_norm_stats.pkl'. Menggunakan default.")
            # Default fallback agar tidak crash (gunakan nilai kira-kira)
            return {
                'MEAN_MAP': 78.5, 'STD_MAP': 16.0, 'IDX_MAP': 6,
                'MEAN_BAL': 1500.0, 'STD_BAL': 3500.0, 'IDX_BAL': 30
            }

    def set_autoencoder(self, ae_model):
        self.autoencoder = ae_model.to(self.device)
        self.autoencoder.eval()

    def calculate_clinical_penalty(self, raw_state, predicted_action):
        """
        Menghitung pinalti klinis. Kompatibel dengan:
        1. Stacked AE (Input 3D: Batch, Seq, Feat)
        2. Standard AE (Input 2D: Batch, Feat)
        """
        
        # --- 1. DETEKSI DIMENSI OTOMATIS ---
        if raw_state.ndim == 3:
            # Kasus Stacked AE: (Batch, 5, 37) -> Ambil jam terakhir
            current_state = raw_state[:, -1, :] 
        else:
            # Kasus AE Biasa: (Batch, 37) -> Pakai langsung
            current_state = raw_state

        # --- 2. AMBIL STATISTIK NORMALISASI ---
        MEAN_MAP = self.norm_stats['MEAN_MAP']
        STD_MAP = self.norm_stats['STD_MAP']
        idx_map = int(self.norm_stats['IDX_MAP'])
        
        MEAN_BAL = self.norm_stats['MEAN_BAL']
        STD_BAL = self.norm_stats['STD_BAL']
        idx_bal = int(self.norm_stats['IDX_BAL'])
        
        # --- 3. DENORMALISASI (Z-Score -> Nilai Asli) ---
        # Pastikan idx_map & idx_bal valid dalam range fitur (0-36)
        map_mmhg = (current_state[:, idx_map] * STD_MAP) + MEAN_MAP
        bal_ml   = (current_state[:, idx_bal] * STD_BAL) + MEAN_BAL
        
        # --- 4. AMBIL AKSI PREDIKSI ---
        action_iv = predicted_action[:, 0]   
        action_vaso = predicted_action[:, 1] 

        # --- 5. RULE 1: VASOPRESSOR SAFETY (MAP > 78 mmHg) ---
        # Jika MAP > 78, stop naikkan vaso.
        # ambang batas 78 diambil sedikit di atas target 65-75 (SSC Guidelines)
        high_bp_mask = (map_mmhg > 78.0).float()
        
        # Hitung seberapa parah kelebihannya (dibagi 10 untuk scaling loss)
        vaso_excess = torch.relu(map_mmhg - 78.0) / 10.0
        
        # Pinalti aktif jika MAP tinggi DAN Agen memberikan dosis vaso (> -0.5)
        vaso_penalty = high_bp_mask * vaso_excess * torch.relu(action_vaso + 0.5)

        # --- 6. RULE 2: FLUID SAFETY (Balance > 5000 mL) ---
        # Jika akumulasi cairan > 5L, stop guyur cairan.
        overload_mask = (bal_ml > 5000.0).float()
        
        # Scaling per 1000 mL (1 Liter)
        fluid_excess = torch.relu(bal_ml - 5000.0) / 1000.0
        
        # Pinalti aktif jika Overload DAN Agen memberikan cairan (> -0.5)
        fluid_penalty = overload_mask * fluid_excess * torch.relu(action_iv + 0.5)
        
        # --- 7. RETURN TOTAL PENALTY (MEAN BATCH) ---
        return (vaso_penalty + fluid_penalty).mean()

    def train(self, batches, epoch):
        # 1. Unpack Data
        (state, next_state, action, next_action,
         reward, done, bloc_num, SOFAS) = batches

        # 2. Pindahkan semua ke GPU
        state = state.clone().detach().float().to(self.device)
        next_state = next_state.clone().detach().float().to(self.device)
        action = action.clone().detach().float().to(self.device)
        reward = reward.clone().detach().float().to(self.device)
        done = done.clone().detach().float().to(self.device)
        bloc_num = torch.tensor(bloc_num).long().to(self.device)

        batch_size = 128
        uids = torch.unique(bloc_num)
        shuffled_indices = torch.randperm(len(uids))
        uids = uids[shuffled_indices]
        num_batches = len(uids) // batch_size
        
        record_loss = []
        rec_rewards = []
        
        for batch_idx in range(num_batches + 1):
            batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_mask = torch.isin(bloc_num, batch_uids)

            # A. AMBIL RAW DATA (Wajib 37 dimensi)
            batch_state_raw = state[batch_mask]       # <-- Ganti nama jadi _raw
            batch_next_state_raw = next_state[batch_mask]
            
            batch_action = action[batch_mask] 
            batch_reward = reward[batch_mask].unsqueeze(1)
            batch_done = done[batch_mask].unsqueeze(1)

            # B. ENCODING KE LATENT (Untuk Masuk Neural Network)
            if self.autoencoder:
                with torch.no_grad():
                    # Encode Raw -> Latent (24 dim)
                    batch_state_latent = self.autoencoder.encode(batch_state_raw)
                    batch_next_state_latent = self.autoencoder.encode(batch_next_state_raw)
            else:
                # Fallback untuk Standard AE tanpa encoder class
                batch_state_latent = batch_state_raw[:, -1, :] if batch_state_raw.ndim == 3 else batch_state_raw
                batch_next_state_latent = batch_next_state_raw[:, -1, :] if batch_next_state_raw.ndim == 3 else batch_next_state_raw

            # ================= CRITIC UPDATE =================
            with torch.no_grad():
                # Gunakan LATENT untuk Actor/Critic
                next_action_new, next_log_prob, _, _ = self.actor.sample(batch_next_state_latent)
                target_q1 = self.target_critic_1(batch_next_state_latent, next_action_new)
                target_q2 = self.target_critic_2(batch_next_state_latent, next_action_new)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                q_target = batch_reward + (1 - batch_done) * self.gamma * target_q
                
            q1 = self.critic_1(batch_state_latent, batch_action)
            q2 = self.critic_2(batch_state_latent, batch_action)
            critic_1_loss = F.mse_loss(q1, q_target)
            critic_2_loss = F.mse_loss(q2, q_target)

            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
            self.critic_2_optimizer.step()

            # ================= ACTOR UPDATE =================
            # Gunakan LATENT untuk sample action
            new_action, log_prob, _, _ = self.actor.sample(batch_state_latent)
            
            q1_new = self.critic_1(batch_state_latent, new_action)
            q2_new = self.critic_2(batch_state_latent, new_action)
            
            sac_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
            bc_loss = F.mse_loss(new_action, batch_action)
            
            # --- FIX: Gunakan RAW STATE untuk hitung pinalti ---
            # batch_state_raw ukurannya (B, 37) atau (B, 5, 37), aman untuk index 27
            safety_penalty = self.calculate_clinical_penalty(batch_state_raw, new_action)
            
            actor_loss = sac_loss + (self.bc_weight * bc_loss) + (self.safety_weight * safety_penalty)

            rec_rewards.append(batch_reward.detach().cpu().numpy())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # ================= ALPHA UPDATE =================
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            self.soft_update()

            avg_loss = (critic_1_loss + critic_2_loss + actor_loss).item() / 3
            record_loss.append(avg_loss)

            if batch_idx % 25 == 0:
                if len(record_loss) > 0:
                    curr_loss = np.mean(record_loss)
                    self.actor_scheduler.step(curr_loss)
                    self.critic_1_scheduler.step(curr_loss)
                    self.critic_2_scheduler.step(curr_loss)
                    self.alpha_scheduler.step(curr_loss)
                
                temp_rewards = np.concatenate(rec_rewards).squeeze()
                est_reward = np.mean(temp_rewards)
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {avg_loss:.4f}, Safety: {safety_penalty.item():.4f}, Rew: {est_reward:.2f}")

        return record_loss
        
    def soft_update(self):
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, state, deterministic=True):
        state = torch.tensor(state).float().to(self.device)
        if state.ndim == 1: state = state.unsqueeze(0)
        
        # Encode jika ada AE
        if self.autoencoder:
            with torch.no_grad():
                state = self.autoencoder.encode(state)

        with torch.no_grad():
            mean, std = self.actor(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = torch.tanh(dist.sample())
        return action.cpu().numpy()
# --- 4. FIXED ENSEMBLE SAC ---
class EnsembleSAC:
    # Update init arguments
    def __init__(self, num_agents=5, state_dim=24, action_dim=2, 
                 bc_weight=0.1, safety_weight=1.0,
                 action_space_min=None, action_space_max=None): # <--- PARAMETER BARU
        
        self.num_agents = num_agents
        # Teruskan ke SACAgent
        self.agents = [
            SACAgent(state_dim, action_dim, bc_weight=bc_weight, safety_weight=safety_weight,
                     action_space_min=action_space_min, action_space_max=action_space_max) 
            for _ in range(num_agents)
        ]
    def set_autoencoder(self, autoencoder):
        for agent in self.agents:
            agent.set_autoencoder(autoencoder)

    def train(self, batches, epoch):
        # Idealnya, setiap agent mendapatkan "Bootstrapped" data (subset berbeda dari batch)
        # Tapi untuk simpelnya, kita latih pada batch yang sama dulu
        losses = []
        for agent in self.agents:
            l = agent.train(batches, epoch)
            losses.append(l)
        return losses

    def get_action(self, state, strategy='vote'):
        # state: shape [batch, input_dim] atau [input_dim]
        # Pastikan input numpy array
        if isinstance(state, list): state = np.array(state)
        
        # Kumpulkan prediksi dari semua agent
        # actions shape: (num_agents, batch_size, action_dim)
        all_actions = [] 
        all_q_values = [] # Untuk strategi vote
        
        # Konversi state ke tensor sekali saja untuk perhitungan Q-Value
        state_tensor = torch.tensor(state).float().to(device)
        if state_tensor.ndim == 1: state_tensor = state_tensor.unsqueeze(0)
        
        if hasattr(self.agents[0], 'autoencoder') and self.agents[0].autoencoder:
             state_tensor = self.agents[0].autoencoder.encode(state_tensor)

        for agent in self.agents:
            # Get Action
            action_batch = agent.get_action(state, deterministic=True) 
            all_actions.append(action_batch)
            
            if strategy == 'vote':
                with torch.no_grad():
                    act_t = torch.tensor(action_batch).float().to(device)
                    q1 = agent.critic_1(state_tensor, act_t)
                    q2 = agent.critic_2(state_tensor, act_t)
                    q_min = torch.min(q1, q2) # Pessimistic Q
                    all_q_values.append(q_min.cpu().numpy())

        all_actions = np.array(all_actions) # (Agents, Batch, Dim)
        
        if strategy == 'mean':
            # Rata-rata action dari semua agent
            return np.mean(all_actions, axis=0) # Output: (Batch, Dim)
        
        elif strategy == 'vote':
            # Vote berdasarkan Q-value tertinggi (Pessimistic Q Max)
            # all_q_values shape: (Agents, Batch, 1)
            all_q_values = np.array(all_q_values)
            
            # Cari index agent dengan Q value tertinggi untuk setiap sample dalam batch
            # argmax pada axis 0 (antar agent)
            best_agent_idxs = np.argmax(all_q_values, axis=0) # Shape: (Batch, 1)
            
            final_actions = []
            batch_size = all_actions.shape[1]
            for i in range(batch_size):
                idx = best_agent_idxs[i, 0]
                final_actions.append(all_actions[idx, i, :])
                
            return np.array(final_actions)

        else:
            raise ValueError("Unknown Strategy")