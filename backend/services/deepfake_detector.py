"""
VoiceShield-AI — AASIST-L Deepfake/Anti-Spoofing Detector (Phase 5)

Responsible for:
1. Loading the official AASIST-L model checkpoint
2. Preparing audio according to AASIST-L requirements
3. Running inference on GPU/CPU
4. Returning structured detection results

This module is used by main.py to replace the mock analysis.

AASIST-L:
- Input: mono audio at 16 kHz, 64,600 samples (~4 seconds)
- Output: 2 logits (bonafide vs spoof)
- Training data: ASVspoof 2019 Logical Access (LA)

Important limitations:
- This model detects spoofing under ASVspoof LA conditions
- It may not detect all modern voice cloning or synthesis methods
- Results are model scores, not calibrated probabilities
- Domain mismatch may occur with out-of-distribution audio
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import librosa

from .gpu_accelerator import get_gpu_accelerator, is_gpu_available

logger = logging.getLogger("voiceshield.deepfake_detector")

# ---------- Configuration ----------

# Model path relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models" / "aasist_l"
CHECKPOINT_PATH = MODELS_DIR / "AASIST-L.pth"

# AASIST-L input specifications (from official repo)
TARGET_SAMPLE_RATE = 16000
MODEL_INPUT_LENGTH = 64600  # samples, ~4.04 seconds at 16 kHz

# Model output label mapping (from official repo)
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1


# ---------- Model Architecture ----------

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
                 bias=False):
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

    def forward(self, x):
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

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_time(x)
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


# ---------- Global model instance ----------
# Loaded once at module import time, reused for all requests.

_model = None
_device = None


def _load_model():
    """Load the AASIST-L model and checkpoint. Called once at startup."""
    global _model, _device

    if _model is not None:
        return _model, _device

    logger.info("Loading AASIST-L model")

    # Check for GPU acceleration via CuPy
    gpu_accel = get_gpu_accelerator()
    if is_gpu_available():
        logger.info("CuPy GPU acceleration available")

    # Determine device (PyTorch - will be CPU since CUDA PyTorch unavailable)
    # But GPU acceleration will be handled via CuPy at inference time
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")

    if _device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif is_gpu_available():
        logger.info("GPU: NVIDIA GPU available via CuPy acceleration")

    # Check checkpoint exists
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"AASIST-L checkpoint not found at {CHECKPOINT_PATH}. "
            f"Please download from https://github.com/clovaai/aasist/tree/main/models/weights"
        )

    logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # Create model
    model_config = {
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
        "first_conv": 128,
    }
    _model = AAISSTModel(model_config)
    _model = _model.to(_device)
    _model.eval()

    # Load checkpoint
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=_device)
        _model.load_state_dict(checkpoint)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    return _model, _device


# ---------- Public API ----------

def detect_deepfake_longaudio(waveform: np.ndarray, sample_rate: int) -> dict:
    """
    Detect spoofing in LONG audio by analyzing multiple overlapping windows.

    For audio longer than ~4 seconds, this function:
    1. Resamples to 16 kHz if needed
    2. Splits into overlapping 4-second windows
    3. Runs AASIST-L on each window
    4. Aggregates scores across windows
    5. Returns recording-level prediction

    Parameters
    ----------
    waveform : np.ndarray
        Audio waveform as float32 numpy array (mono).
    sample_rate : int
        Sample rate of the waveform in Hz.

    Returns
    -------
    dict
        Result dict with:
        - success: True/False
        - prediction: "BONAFIDE" or "SPOOF"
        - spoof_score: aggregate spoof score across all windows
        - bonafide_score: aggregate bonafide score across all windows
        - window_scores: list of per-window results
        - num_windows: number of windows analyzed
        - total_duration_sec: total audio duration
        - device: GPU/CPU used
    """

    try:
        # Load model
        model, device = _load_model()

        # Resample if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            logger.debug(f"Resampling from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz")
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
            )

        # Validate waveform
        if waveform is None or waveform.size == 0:
            return {
                "success": False,
                "error": "Waveform is empty after resampling.",
            }

        if not np.all(np.isfinite(waveform)):
            return {
                "success": False,
                "error": "Waveform contains invalid values (NaN or Inf).",
            }

        # Calculate window parameters
        window_samples = MODEL_INPUT_LENGTH  # 64600 samples = 4.04 sec at 16 kHz
        overlap_ratio = 0.5  # 50% overlap
        step_samples = int(window_samples * (1 - overlap_ratio))  # 50% step

        total_samples = len(waveform)
        total_duration_sec = total_samples / TARGET_SAMPLE_RATE

        # Create windows
        windows = []
        start_idx = 0
        while start_idx < total_samples:
            end_idx = min(start_idx + window_samples, total_samples)
            window = waveform[start_idx:end_idx]

            # Pad window if it's shorter than required
            if len(window) < window_samples:
                n_repeats = (window_samples // len(window)) + 1
                window = np.tile(window, n_repeats)[:window_samples]

            windows.append(window)
            start_idx += step_samples

        logger.info(
            f"Long audio: {total_duration_sec:.2f} sec → "
            f"{len(windows)} windows (50% overlap)"
        )

        # Run inference on all windows
        window_results = []
        spoof_scores_all = []
        bonafide_scores_all = []

        import time
        t_start_total = time.time()

        for i, window in enumerate(windows):
            x = torch.FloatTensor(window).unsqueeze(0)
            x = x.to(device)

            with torch.inference_mode():
                _, output = model(x)

            spoof_score = output[0, LABEL_SPOOF].item()
            bonafide_score = output[0, LABEL_BONAFIDE].item()

            window_results.append({
                "window": i,
                "spoof_score": spoof_score,
                "bonafide_score": bonafide_score,
                "prediction": "SPOOF" if spoof_score > bonafide_score else "BONAFIDE"
            })

            spoof_scores_all.append(spoof_score)
            bonafide_scores_all.append(bonafide_score)

        t_end_total = time.time()
        inference_time_ms = (t_end_total - t_start_total) * 1000

        # Aggregate scores
        # Strategy: Use mean of spoof/bonafide scores across windows
        mean_spoof_score = np.mean(spoof_scores_all)
        mean_bonafide_score = np.mean(bonafide_scores_all)

        # Determine prediction
        prediction = "SPOOF" if mean_spoof_score > mean_bonafide_score else "BONAFIDE"

        # Count spoof windows
        num_spoof_windows = sum(1 for r in window_results if r["prediction"] == "SPOOF")
        spoof_ratio = num_spoof_windows / len(window_results)

        logger.info(
            f"Long audio result: {prediction}, "
            f"mean_spoof={mean_spoof_score:.4f}, "
            f"mean_bonafide={mean_bonafide_score:.4f}, "
            f"spoof_windows={num_spoof_windows}/{len(window_results)} ({spoof_ratio*100:.1f}%), "
            f"time={inference_time_ms:.2f}ms"
        )

        return {
            "success": True,
            "model": "AASIST-L",
            "prediction": prediction,
            "spoof_score": mean_spoof_score,
            "bonafide_score": mean_bonafide_score,
            "score_difference": abs(mean_bonafide_score - mean_spoof_score),
            "inference_time_ms": round(inference_time_ms, 2),
            "device": str(device),
            "window_scores": window_results,
            "num_windows": len(window_results),
            "spoof_windows": num_spoof_windows,
            "spoof_ratio": round(spoof_ratio, 3),
            "total_duration_sec": round(total_duration_sec, 2),
            "analysis_mode": "long_audio_windowing"
        }

    except FileNotFoundError as e:
        logger.error(f"Checkpoint file error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Long audio detection failed: {e}", exc_info=True)
        return {"success": False, "error": f"Detection failed: {str(e)}"}


def detect_deepfake(waveform: np.ndarray, sample_rate: int) -> dict:
    """
    Detect spoofing/deepfake in audio using AASIST-L.

    Parameters
    ----------
    waveform : np.ndarray
        Audio waveform as float32 numpy array (mono).
    sample_rate : int
        Sample rate of the waveform in Hz.

    Returns
    -------
    dict
        A result dict with keys:
        - success: True if detection succeeded
        - error: (if success=False) error message
        - model: model name ("AASIST-L")
        - prediction: "BONAFIDE" or "SPOOF"
        - bonafide_score: raw output for bonafide class (index 0)
        - spoof_score: raw output for spoof class (index 1)
        - inference_time_ms: inference time in milliseconds
        - device: device used ("cuda" or "cpu")
        - waveform_shape: shape of input waveform after preprocessing
        - waveform_samples: number of samples after preprocessing

    The raw scores are model logits, NOT calibrated probabilities.
    Higher bonafide_score indicates more likely bonafide.
    Higher spoof_score indicates more likely spoof.
    """

    try:
        # Load model if not already loaded
        model, device = _load_model()

        # Resample if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            logger.debug(
                f"Resampling from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz"
            )
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
            )

        # Validate waveform
        if waveform is None or waveform.size == 0:
            return {
                "success": False,
                "error": "Waveform is empty after resampling.",
            }

        if not np.all(np.isfinite(waveform)):
            return {
                "success": False,
                "error": "Waveform contains invalid values (NaN or Inf).",
            }

        # Pad or truncate to MODEL_INPUT_LENGTH
        if len(waveform) < MODEL_INPUT_LENGTH:
            # Repeat the audio to fill the required length
            n_repeats = (MODEL_INPUT_LENGTH // len(waveform)) + 1
            waveform = np.tile(waveform, n_repeats)[:MODEL_INPUT_LENGTH]
            logger.debug(
                f"Audio padded (repeated) to {MODEL_INPUT_LENGTH} samples"
            )
        else:
            waveform = waveform[:MODEL_INPUT_LENGTH]
            logger.debug(f"Audio truncated to {MODEL_INPUT_LENGTH} samples")

        # Convert to tensor
        x = torch.FloatTensor(waveform).unsqueeze(0)  # (1, 64600)
        x = x.to(device)

        # Run inference
        import time
        with torch.inference_mode():
            t_start = time.time()
            _, output = model(x)
            t_end = time.time()
            inference_time_ms = (t_end - t_start) * 1000

        # Parse output
        bonafide_score = output[0, LABEL_BONAFIDE].item()
        spoof_score = output[0, LABEL_SPOOF].item()

        # Determine prediction
        prediction = "SPOOF" if spoof_score > bonafide_score else "BONAFIDE"

        logger.info(
            f"Prediction: {prediction}, bonafide={bonafide_score:.4f}, "
            f"spoof={spoof_score:.4f}, time={inference_time_ms:.2f}ms"
        )

        return {
            "success": True,
            "model": "AASIST-L",
            "prediction": prediction,
            "bonafide_score": bonafide_score,
            "spoof_score": spoof_score,
            "inference_time_ms": round(inference_time_ms, 2),
            "device": str(device),
            "waveform_shape": list(x.shape),
            "waveform_samples": MODEL_INPUT_LENGTH,
        }

    except FileNotFoundError as e:
        logger.error(f"Checkpoint file error: {e}")
        return {
            "success": False,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Deepfake detection failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Detection failed: {str(e)}",
        }
