
import pandas as pd
import os
import numpy as np
import pandas as pd
import numpy as np
import os
from SAC_deepQnet import EnsembleSAC, AutoEncoder
import pandas as pd
import torch
import numpy as np
import pickle
from treatment_recommendation_service import physicianAction, aiRecommendation


# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

physpol = np.load(os.path.join(BASE_DIR, 'model/phys_actionsb.npy'))
data_file = os.path.join(BASE_DIR, 'df_with_readable_charttime.csv')
model_path = os.path.join(BASE_DIR, 'model/best_agent_ensemble.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def csv_to_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    return df

def load_model(model_path, device):
    """
    Memuat model Ensemble SAC + Temporal VAE yang sudah disimpan untuk deployment.
    
    Args:
        model_path (str): Path ke file .pt (misal: 'SACEnsemble-algorithm/best_agent_ensemble.pt')
        device (torch.device): CPU atau CUDA
        
    Returns:
        ensemble (EnsembleSAC): Model siap pakai (sudah berisi VAE & weight terlatih)
    """

    print(f"\n[Loader] Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")

    # ==============================================================================
    # 1. SETUP & LOAD AUTOENCODER
    # ==============================================================================
    LATENT_DIM = 24  # Harus sama dengan output encoder yg dilatih sebelumnya
    INPUT_DIM = 37   # Raw feature dimension
    NUM_AGENTS = 5
    ACTION_DIM = 2
    BC_WEIGHT = 0.25   # Tidak berpengaruh saat inferensi, tapi butuh untuk init

    # Inisialisasi arsitektur AE (pastikan class AutoEncoder sudah didefinisikan di atas)
    ae_model = AutoEncoder(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=LATENT_DIM).to(device)

    # Load bobot yang sudah dilatih (dari tahap sebelumnya)
    try:
        ae_path = os.path.join(BASE_DIR, 'model/best_ae_mimic.pth')
        ae_model.load_state_dict(
            torch.load(ae_path, map_location=device)
        )
        print("✓ Pre-trained AutoEncoder loaded successfully.")
    except FileNotFoundError:
        print("! WARNING: 'best_ae_mimic.pth' not found. Will use weights from checkpoint.")

    # Bekukan AutoEncoder (Freeze) agar tidak berubah saat training RL
    for param in ae_model.parameters():
        param.requires_grad = False
    ae_model.eval()

    
    # B. Inisialisasi Ensemble SAC
    # PENTING: state_dim harus VAE_LATENT_DIM (64), bukan raw 37
    ensemble = EnsembleSAC(
        num_agents=NUM_AGENTS, 
        state_dim=LATENT_DIM, 
        action_dim=ACTION_DIM, 
        bc_weight=BC_WEIGHT
    )
    print(ensemble)

    # --- 3. LOAD WEIGHTS ---
    try:
        # Load checkpoint ke device yang benar
        checkpoint = torch.load(model_path, map_location=device)
        
        # A. Load VAE Weights
        ae_model.load_state_dict(checkpoint['autoencoder_state_dict'])
        print("Temporal AE weights loaded.")

        # B. Load Ensemble Weights (Actor & Critic)
        # Checkpoint menyimpan list of state_dicts
        actor_dicts = checkpoint['actor_state_dicts']
        critic1_dicts = checkpoint['critic1_state_dicts']
        critic2_dicts = checkpoint['critic2_state_dicts']

        for i, agent in enumerate(ensemble.agents):
            agent.actor.load_state_dict(actor_dicts[i])
            agent.critic_1.load_state_dict(critic1_dicts[i])
            agent.critic_2.load_state_dict(critic2_dicts[i])
            
            # Pindahkan agent ke device
            agent.actor.to(device)
            agent.critic_1.to(device)
            agent.critic_2.to(device)
            
        print(f"✓ Ensemble weights loaded for {len(ensemble.agents)} agents.")
        
        # Metadata check (optional)
        if 'best_mean_agent_q' in checkpoint:
            print(f"  > Best Validation Q-Value recorded: {checkpoint['best_mean_agent_q']:.4f}")

    except KeyError as e:
        print(f"! ERROR: Struktur file model tidak cocok. Key hilang: {e}")
        return None
    except Exception as e:
        print(f"! ERROR Loading Model: {e}")
        return None

    # --- 4. INTEGRASI & FREEZE ---
    # Masukkan VAE ke dalam Ensemble
    ensemble.set_autoencoder(ae_model)
    
    # Set mode evaluasi (Matikan Dropout, Batchnorm statistik beku)
    ae_model.eval()
    for agent in ensemble.agents:
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()
        
    # Freeze Gradients (Hemat memori saat inferensi)
    for param in ae_model.parameters(): param.requires_grad = False
    for agent in ensemble.agents:
        for param in agent.actor.parameters(): param.requires_grad = False
        for param in agent.critic_1.parameters(): param.requires_grad = False
        for param in agent.critic_2.parameters(): param.requires_grad = False

    print("[Loader] Model ready for deployment.")
    return ensemble


class App():
    def __init__(self):
        self.df = csv_to_dataframe(data_file)
        self.model = load_model(model_path, device)
        self.stats_path = os.path.join(BASE_DIR, 'model/action_norm_stats.pkl')

    def get_patient_list(self):
        """
        This function extract all patient id from the dataframe
        @output: list of patient id (list of int)
        """
        df = self.df
        patient_list = df['icustayid'].unique().tolist()
        return patient_list

    def get_hr(self,patient_id:int):
        """
        This function extract all heart rate states from all blocs of the patient and return as an list
        example: [74, 83, ...]

        @param patient_id : id of patient as an string
        @output: array of heart rate states (list of float)
        """
        df = self.df

        patient_df = df[df['icustayid'] == patient_id] #filter other patient data
        hr_df = patient_df['HR'].to_list()

        print(f"TEST: type of hr list: {type(hr_df[0])}")

    def get_spo2(self, patient_id:int):
        """
        This function extract all  oxygen saturation (sp02) states from all blocs of the patient and return as an list
        example: [99.1, 92.0, ...]

        @param patient_id : id of patient as an string
        @output: array of Spo2 states (list of float)
        """
        df = self.df

        patient_df = df[df['icustayid'] == patient_id] #filter other patient data
        spo2_df = patient_df['SpO2'].to_list()

        print(f"TEST: type of spo2 list: {type(spo2_df[0])}")

    def get_rr(self, patient_id:int):
        """
        This function extract all respiratory rate states (RR) from all blocs of the patient and return as an list
        example: [18.8, 26.5, ...]

        @param patient_id : id of patient as an string
        @output: array of RR states (list of float)
        """
        df = self.df

        patient_df = df[df['icustayid'] == patient_id] #filter other patient data
        rr_df = patient_df['RR'].to_list()
        print(f"TEST: type of rr list: {type(rr_df[0])}")

    def get_bp(self, patient_id):
        """
        This function extract all blood preasure states (BP) from all blocs of the patient and return as an list of tuple
        (sys_bp, dia_bp)
        example: [(130, 50), (120, 45), ...]

        @param patient_id : id of patient as an string
        @output: array of BP states (list of tuple containing float)
        """
        df = self.df

        patient_df = df[df['icustayid'] == patient_id] #filter other patient data
        sys_bp = patient_df['SysBP'].to_list()
        dia_bp = patient_df['DiaBP'].to_list()

        bp_list = zip(sys_bp, dia_bp)
        return bp_list
    
    def get_temp(self, patient_id):
        """
        This function extract all temperature states (Temp) from all blocs of the patient and return as an list
        example: [36.5, 37.0, ...]

        @param patient_id : id of patient as an string
        @output: array of Temp states (list of float)
        """
        df = self.df
        patient_df = df[df['icustayid'] == patient_id] #filter other patient data
        temp_df = patient_df['Temp_C'].to_list()
        print(f"TEST: type of temp list: {type(temp_df[0])}")

    def predict(self, user_input):
        try:
            
            # loaded_model.actor.eval()
            
            # Daftar kolom
            colnorm = ['SOFA', 'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C',
                    'Sodium', 'Chloride', 'Glucose', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count',
                    'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'HCO3', 'Arterial_lactate', 'Shock_Index',
                    'PaO2_FiO2', 'cumulated_balance', 'CO2_mEqL', 'Ionised_Ca']
            collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT',
                    'Total_bili', 'INR', 'input_total', 'output_total']

            # Get available columns from dataframe
            available_colnorm = [c for c in colnorm if c in user_input.columns]
            available_collog = [c for c in collog if c in user_input.columns]

            if len(available_colnorm) > 0:
                reformat_colnorm = np.asarray(user_input[available_colnorm].values, dtype=np.float64)

                # Hardcoded mean dan std
                mean = np.mean(reformat_colnorm)
                std_dev = np.std(reformat_colnorm)
                if std_dev == 0:
                    std_dev = 1.0  # Avoid division by zero

                reformat_colnorm = (reformat_colnorm - mean) / std_dev
            else:
                reformat_colnorm = np.zeros((len(user_input), len(colnorm)))

            if len(available_collog) > 0:
                reformat_collog = np.asarray(user_input[available_collog].values, dtype=np.float64)
                reformat_collog = np.log(0.1 + reformat_collog)
            else:
                reformat_collog = np.zeros((len(user_input), len(collog)))

            processed_state = np.hstack((reformat_colnorm, reformat_collog))

            if np.isnan(processed_state).any():
                raise ValueError("Preprocessed input contains NaNs. Check your data.")

            single_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)
            
            # 1. LOAD STATS DARI FILE DULU
            stats_path = os.path.join(BASE_DIR, 'model/action_norm_stats.pkl')

            # Inference action dari model
            # with torch.no_grad():
            #     norm_action, _, _, _ = loaded_model.sample(single_state)
            #     norm_action = norm_action.cpu().numpy().reshape(1, -1)

            # ===== Tambahkan inverse transform dari z-score ke dosage =====
            def inverse_transform_action(norm_action, stats_path):
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)

                mean_log_iv = stats['mean_log_iv']
                std_log_iv = stats['std_log_iv']
                mean_log_vaso = stats['mean_log_vaso']
                std_log_vaso = stats['std_log_vaso']

                # Transformasi balik dari z-score ke log1p
                iv_log = norm_action[:, 0] * std_log_iv + mean_log_iv
                vaso_log = norm_action[:, 1] * std_log_vaso + mean_log_vaso
                iv_log = np.abs(iv_log)
                vaso_log = np.abs(vaso_log)
                # Transformasi balik ke domain asli
                iv_raw = np.expm1(iv_log)
                vaso_raw = np.expm1(vaso_log)
                return np.stack([iv_raw, vaso_raw], axis=1)
            # Ambil aksi acak dari physpol
            idx = np.random.randint(len(physpol))
            physician_action = physpol[idx]  # normalized action

            # Inverse transform dari normalized → raw
            physician_action = inverse_transform_action(physician_action.reshape(1, -1), stats_path) [0]

            print("physpol shape:", np.array(physpol).shape)
            print("sample physpol[idx]:", physician_action)
            print("physician_action shape:", physician_action.shape)
            
            return physicianAction(physician_action)
        except Exception as e:
            return {"error": str(e)}

    def predict_personalized(self, user_input):
        """
        Generate personalized AI treatment recommendation using the SAC model.
        Unlike predict() which returns random historical physician action,
        this method uses the trained model to optimize treatment for the specific patient state.
        
        @param user_input: DataFrame containing patient state data
        @return: dict with AI-recommended IV and vasopressor doses
        """
        try:
            # Daftar kolom
            colnorm = ['SOFA', 'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C',
                    'Sodium', 'Chloride', 'Glucose', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count',
                    'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'HCO3', 'Arterial_lactate', 'Shock_Index',
                    'PaO2_FiO2', 'cumulated_balance', 'CO2_mEqL', 'Ionised_Ca']
            collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT',
                    'Total_bili', 'INR', 'input_total', 'output_total']

            # Get available columns from dataframe
            available_colnorm = [c for c in colnorm if c in user_input.columns]
            available_collog = [c for c in collog if c in user_input.columns]

            if len(available_colnorm) > 0:
                reformat_colnorm = np.asarray(user_input[available_colnorm].values, dtype=np.float64)

                # Normalize
                mean = np.mean(reformat_colnorm)
                std_dev = np.std(reformat_colnorm)
                if std_dev == 0:
                    std_dev = 1.0

                reformat_colnorm = (reformat_colnorm - mean) / std_dev
            else:
                reformat_colnorm = np.zeros((len(user_input), len(colnorm)))

            if len(available_collog) > 0:
                reformat_collog = np.asarray(user_input[available_collog].values, dtype=np.float64)
                reformat_collog = np.log(0.1 + reformat_collog)
            else:
                reformat_collog = np.zeros((len(user_input), len(collog)))

            processed_state = np.hstack((reformat_colnorm, reformat_collog))

            if np.isnan(processed_state).any():
                raise ValueError("Preprocessed input contains NaNs. Check your data.")

            single_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)

            # Get action from SAC model - personalized prediction
            with torch.no_grad():
                norm_action = self.model.get_action(single_state, strategy='mean')
                norm_action = norm_action.reshape(1, -1)

            # Load stats for inverse transform
            stats_path = os.path.join(BASE_DIR, 'model/action_norm_stats.pkl')
            
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)

            mean_log_iv = stats['mean_log_iv']
            std_log_iv = stats['std_log_iv']
            mean_log_vaso = stats['mean_log_vaso']
            std_log_vaso = stats['std_log_vaso']

            # Inverse transform from z-score to log1p
            iv_log = norm_action[:, 0] * std_log_iv + mean_log_iv
            vaso_log = norm_action[:, 1] * std_log_vaso + mean_log_vaso
            iv_log = np.abs(iv_log)
            vaso_log = np.abs(vaso_log)
            
            # Transform back to original domain
            iv_raw = np.expm1(iv_log)
            vaso_raw = np.expm1(vaso_log)
            raw_action = np.stack([iv_raw, vaso_raw], axis=1)

            print(f"[AI Model] Predicted action: IV={raw_action[0, 0]:.3f} ml, Vaso={raw_action[0, 1]:.3f} ug/kg/min")

            return aiRecommendation(raw_action[0])
        except Exception as e:
            return {"error": str(e)}