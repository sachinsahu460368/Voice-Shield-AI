"""
VoiceShield-AI — AASIST-L Standalone Model Test (Phase 5)

This script:
1. Downloads the official AASIST-L.pth checkpoint
2. Loads the AASIST model architecture
3. Loads a test audio file
4. Runs inference without FastAPI
5. Prints the raw model output and prediction

This is NOT integrated into the FastAPI backend yet.
Used only to verify AASIST-L works correctly.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import librosa

# Add backend to path so we can import the model
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

print("=" * 80)
print("AASIST-L STANDALONE TEST")
print("=" * 80)

# ---------- Configuration ----------

MODEL_DIR = backend_dir / "models" / "aasist_l"
CHECKPOINT_URL = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST-L.pth"
CHECKPOINT_PATH = MODEL_DIR / "AASIST-L.pth"

TEST_AUDIO_PATH = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3")

TARGET_SAMPLE_RATE = 16000
MODEL_INPUT_LENGTH = 64600  # 4.04 seconds at 16 kHz

# ---------- 1. Verify PyTorch / CUDA ----------

print("\n[1] ENVIRONMENT")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ---------- 2. Download checkpoint if needed ----------

print("\n[2] CHECKPOINT MANAGEMENT")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if CHECKPOINT_PATH.exists():
    print(f"[OK] Checkpoint exists: {CHECKPOINT_PATH}")
    checkpoint_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
    print(f"     Size: {checkpoint_size_mb:.2f} MB")
else:
    print(f"Downloading checkpoint from: {CHECKPOINT_URL}")
    print(f"Saving to: {CHECKPOINT_PATH}")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        print(f"[OK] Download complete")
        checkpoint_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        print(f"     Size: {checkpoint_size_mb:.2f} MB")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

# ---------- 3. Define AASIST model architecture ----------

print("\n[3] MODEL ARCHITECTURE")

class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = torch.nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = torch.nn.Linear(in_dim, out_dim)
        self.proj_without_att = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.input_drop = torch.nn.Dropout(p=0.2)
        self.act = torch.nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = torch.nn.functional.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = torch.nn.Linear(in_dim, in_dim)
        self.proj_type2 = torch.nn.Linear(in_dim, in_dim)
        self.att_proj = torch.nn.Linear(in_dim, out_dim)
        self.att_projM = torch.nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = torch.nn.Linear(in_dim, out_dim)
        self.proj_without_att = torch.nn.Linear(in_dim, out_dim)
        self.proj_with_attM = torch.nn.Linear(in_dim, out_dim)
        self.proj_without_attM = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.input_drop = torch.nn.Dropout(p=0.2)
        self.act = torch.nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = torch.nn.functional.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)
        att_map = att_board
        att_map = att_map / self.temp
        att_map = torch.nn.functional.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out


class GraphPool(torch.nn.Module):
    def __init__(self, k: float, in_dim: int, p: float):
        super().__init__()
        self.k = k
        self.sigmoid = torch.nn.Sigmoid()
        self.proj = torch.nn.Linear(in_dim, 1)
        self.drop = torch.nn.Dropout(p=p) if p > 0 else torch.nn.Identity()

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class SincConv(torch.nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 in_channels=1, stride=1, padding=0, dilation=1,
                 bias=False, groups=1, mask=False):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only support one input channel")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = torch.Tensor(np.hamming(
                self.kernel_size)) * torch.Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        self.filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return torch.nn.functional.conv1d(x, self.filters, stride=self.stride,
                                          padding=self.padding, dilation=self.dilation)


class ResidualBlock(torch.nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = torch.nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = torch.nn.Conv2d(in_channels=nb_filts[0],
                                     out_channels=nb_filts[1],
                                     kernel_size=(2, 3),
                                     padding=(1, 1),
                                     stride=1)
        self.selu = torch.nn.SELU(inplace=True)
        self.bn2 = torch.nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = torch.nn.Conv2d(in_channels=nb_filts[1],
                                     out_channels=nb_filts[1],
                                     kernel_size=(2, 3),
                                     padding=(0, 1),
                                     stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = torch.nn.Conv2d(in_channels=nb_filts[0],
                                                   out_channels=nb_filts[1],
                                                   padding=(0, 1),
                                                   kernel_size=(1, 3),
                                                   stride=1)
        else:
            self.downsample = False
        self.mp = torch.nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        out = self.mp(out)
        return out


class AAISSTModel(torch.nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.conv_time = SincConv(out_channels=filts[0],
                                  kernel_size=d_args["first_conv"],
                                  in_channels=1)
        self.first_bn = torch.nn.BatchNorm2d(num_features=1)
        self.drop = torch.nn.Dropout(0.5, inplace=True)
        self.drop_way = torch.nn.Dropout(0.2, inplace=True)
        self.selu = torch.nn.SELU(inplace=True)

        self.encoder = torch.nn.Sequential(
            torch.nn.Sequential(ResidualBlock(nb_filts=filts[1], first=True)),
            torch.nn.Sequential(ResidualBlock(nb_filts=filts[2])),
            torch.nn.Sequential(ResidualBlock(nb_filts=filts[3])),
            torch.nn.Sequential(ResidualBlock(nb_filts=filts[4])),
            torch.nn.Sequential(ResidualBlock(nb_filts=filts[4])),
            torch.nn.Sequential(ResidualBlock(nb_filts=filts[4])))

        self.pos_S = torch.nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = torch.nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = torch.nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = torch.nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, Freq_aug=False):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = torch.nn.functional.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.encoder(x)

        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output


print("[OK] Model architecture loaded")

# ---------- 4. Load checkpoint ----------

print("\n[4] MODEL LOADING")
print(f"Loading checkpoint: {CHECKPOINT_PATH}")

model_args = {
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
    "first_conv": 128,
}

model = AAISSTModel(model_args)
model = model.to(device)
model.eval()

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    print("[OK] Checkpoint loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load checkpoint: {e}")
    sys.exit(1)

# ---------- 5. Load and preprocess audio ----------

print("\n[5] AUDIO LOADING & PREPROCESSING")
print(f"Loading audio: {TEST_AUDIO_PATH}")

if not TEST_AUDIO_PATH.exists():
    print(f"[ERROR] Audio file not found: {TEST_AUDIO_PATH}")
    sys.exit(1)

try:
    waveform, sr = librosa.load(TEST_AUDIO_PATH, sr=TARGET_SAMPLE_RATE, mono=True)
    print(f"[OK] Audio loaded")
    print(f"     Original sample rate: {sr} Hz")
    print(f"     Duration: {len(waveform) / sr:.2f} seconds")
    print(f"     Waveform shape: {waveform.shape}")
    print(f"     Waveform dtype: {waveform.dtype}")
except Exception as e:
    print(f"[ERROR] Failed to load audio: {e}")
    sys.exit(1)

# Pad or truncate to MODEL_INPUT_LENGTH
if len(waveform) < MODEL_INPUT_LENGTH:
    # Repeat the audio
    n_repeats = (MODEL_INPUT_LENGTH // len(waveform)) + 1
    waveform = np.tile(waveform, n_repeats)[:MODEL_INPUT_LENGTH]
    print(f"[OK] Audio padded (repeated) to {MODEL_INPUT_LENGTH} samples")
else:
    waveform = waveform[:MODEL_INPUT_LENGTH]
    print(f"[OK] Audio truncated to {MODEL_INPUT_LENGTH} samples")

print(f"     Final waveform shape: {waveform.shape}")

# Convert to tensor
x = torch.FloatTensor(waveform).unsqueeze(0)  # (1, 64600)
x = x.to(device)

print(f"[OK] Tensor created")
print(f"     Input shape: {x.shape}")
print(f"     Input dtype: {x.dtype}")
print(f"     Device: {x.device}")

# ---------- 6. Run inference ----------

print("\n[6] INFERENCE")

with torch.inference_mode():
    t_start = time.time()
    last_hidden, output = model(x)
    t_end = time.time()
    inference_time_ms = (t_end - t_start) * 1000

print(f"[OK] Inference complete")
print(f"     Inference time: {inference_time_ms:.2f} ms")
print(f"     Output shape: {output.shape}")
print(f"     Output dtype: {output.dtype}")

# ---------- 7. Parse output ----------

print("\n[7] MODEL OUTPUT & PREDICTION")

# output shape: (1, 2)
# output[0, 0] = bonafide score
# output[0, 1] = spoof score

bonafide_score = output[0, 0].item()
spoof_score = output[0, 1].item()

print(f"\nRaw model output:")
print(f"     output[0]: {output[0].cpu().numpy()}")
print(f"     bonafide_score (index 0): {bonafide_score:.6f}")
print(f"     spoof_score (index 1): {spoof_score:.6f}")

# Determine prediction
# Higher spoof_score means more likely spoof
prediction = "SPOOF" if spoof_score > bonafide_score else "BONAFIDE"

print(f"\nPrediction:")
print(f"     Model prediction: {prediction}")
print(f"     Bonafide score: {bonafide_score:.6f}")
print(f"     Spoof score: {spoof_score:.6f}")

# Compute softmax probabilities for reference (not calibrated)
with torch.inference_mode():
    probs = torch.nn.functional.softmax(output, dim=1)

bonafide_prob = probs[0, 0].item()
spoof_prob = probs[0, 1].item()

print(f"\nSoftmax probabilities (for reference only, NOT calibrated):")
print(f"     P(bonafide): {bonafide_prob:.4f}")
print(f"     P(spoof): {spoof_prob:.4f}")

# ---------- 8. Summary ----------

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"[OK] Model loaded: AASIST-L")
print(f"[OK] Device: {device}")
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] Audio: {TEST_AUDIO_PATH.name}")
print(f"[OK] Sample rate: {TARGET_SAMPLE_RATE} Hz")
print(f"[OK] Input shape: {x.shape}")
print(f"[OK] Input dtype: float32")
print(f"[OK] Input samples: {MODEL_INPUT_LENGTH}")
print(f"[OK] Inference time: {inference_time_ms:.2f} ms")
print(f"[OK] Prediction: {prediction}")
print(f"[OK] Bonafide score: {bonafide_score:.6f}")
print(f"[OK] Spoof score: {spoof_score:.6f}")
print("=" * 80)

print("\n[OK] STANDALONE TEST COMPLETED SUCCESSFULLY")
print("     Next step: Integrate into FastAPI backend (Phase 5 Step 9)")
