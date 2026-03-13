import os
import sys
import json
import math
import random
import wave
import time
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the parent directory to the Python path to find the models module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ctm_rl import ContinuousThoughtMachineRL

@dataclass
class Config:
    data_dir: str = "GAMMATONE_64_100"  # relative to examples/
    cache_dir: str = "audio_cache/speaker_split_gammatone_64_100hz"
    output_dir: str = "outputs/ctm_gammatone_asr_experiments"
    sample_rate: int = 16000
    n_bands: int = 64
    frame_hz: int = 100
    win_ms: float = 25.0
    vocab_size: int = 15  # Will be varied in experiments
    silence_duration_ms: int = 200
    seed: int = 0
    d_input: int = 64
    d_model: int = 256
    n_synch_out: int = 32
    synapse_depth: int = 2
    memory_length: int = 10  # Will be varied in experiments
    iterations: int = 2  # Will be varied in experiments
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 6  # Reduced for faster comparison
    words_per_epoch: int = 150  # Reduced for faster comparison  
    lr: float = 5e-5
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    add_noise_train: bool = True
    snr_db: float = 15.0
    warmup_steps: int = 30  # Reduced for faster comparison
    validate_every: int = 15  # More frequent validation for faster feedback
    # Early stopping parameters - more aggressive for comparison
    early_stop_patience: int = 10  # Reduced for faster comparison
    min_improvement: float = 0.01  # Slightly higher threshold for faster stopping
    # Parallel processing
    max_parallel_experiments: int = 2  # Number of experiments to run in parallel

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
    if sampwidth == 2:
        dtype = np.int16
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        data = data / 32768.0
    elif sampwidth == 1:
        dtype = np.uint8
        data = (np.frombuffer(frames, dtype=dtype).astype(np.float32) - 128.0) / 128.0
    else:
        dtype = np.int32
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 2147483648.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, framerate

def resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    duration = len(x) / orig_sr
    tgt_len = int(round(duration * target_sr))
    if tgt_len <= 1 or len(x) <= 1:
        return np.zeros((tgt_len,), dtype=np.float32)
    xp = np.linspace(0, 1, num=len(x), endpoint=False)
    fp = x.astype(np.float32)
    x_new = np.linspace(0, 1, num=tgt_len, endpoint=False)
    return np.interp(x_new, xp, fp).astype(np.float32)

def hann_window(win_length: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / win_length)

def stft_mag_frames(x: np.ndarray, sr: int, n_fft: int, win_length: int, hop_length: int) -> np.ndarray:
    x = x.astype(np.float32)
    window = hann_window(win_length).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - win_length) // hop_length)
    if n_frames <= 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = x[start:start + win_length]
        if len(frame) < win_length:
            pad = np.zeros((win_length - len(frame),), dtype=np.float32)
            frame = np.concatenate([frame, pad])
        frame = frame * window
        fft = np.fft.rfft(frame, n=n_fft)
        mag = np.abs(fft)
        out[i] = mag
    return out

def erb_space(low_freq: float, high_freq: float, n: int) -> np.ndarray:
    def hz2erb(f):
        return 21.4 * np.log10(4.37e-3 * f + 1.0)
    def erb2hz(e):
        return (10**(e / 21.4) - 1.0) / 4.37e-3
    low_e = hz2erb(low_freq)
    high_e = hz2erb(high_freq)
    erbs = np.linspace(low_e, high_e, n)
    return erb2hz(erbs)

def make_gammatone_like_filterbank(sr: int, n_fft: int, n_bands: int, low_hz: float = 50.0) -> np.ndarray:
    n_freq_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freq_bins)
    centers = erb_space(low_hz, sr / 2.0, n_bands)
    fb = np.zeros((n_bands, n_freq_bins), dtype=np.float32)
    for i, c in enumerate(centers):
        left = centers[i - 1] if i > 0 else 0.0
        right = centers[i + 1] if i < n_bands - 1 else sr / 2.0
        l_edge = 0.5 * (left + c)
        r_edge = 0.5 * (c + right)
        asc = (freqs >= l_edge) & (freqs <= c)
        desc = (freqs > c) & (freqs <= r_edge)
        if c > l_edge:
            fb[i, asc] = (freqs[asc] - l_edge) / (c - l_edge)
        if r_edge > c:
            fb[i, desc] = (r_edge - freqs[desc]) / (r_edge - c)
    fb_sum = fb.sum(axis=1, keepdims=True) + 1e-8
    fb /= fb_sum
    return fb

def compute_gammatone_spectrogram(x: np.ndarray, sr: int, n_bands: int, frame_hz: int, win_ms: float) -> np.ndarray:
    hop_length = int(round(sr / frame_hz))
    win_length = int(round(sr * (win_ms / 1000.0)))
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2
    mag = stft_mag_frames(x, sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    fb = make_gammatone_like_filterbank(sr, n_fft, n_bands)
    power = (mag ** 2) @ fb.T
    return power.astype(np.float32)

class FeatureCache:
    def __init__(self, cfg: Config, lex):
        self.cfg = cfg
        self.lex = lex
        os.makedirs(cfg.cache_dir, exist_ok=True)
    def _cache_path(self, word: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"{word}.npy")
    def get(self, word: str, pickle_path: str) -> np.ndarray:
        cp = self._cache_path(word)
        if os.path.exists(cp):
            return np.load(cp)
        
        # Load from pickle file instead of processing audio
        try:
            with open(pickle_path, 'rb') as f:
                features = pickle.load(f)
            
            # Ensure it's the right format (should be gammatone features)
            if isinstance(features, np.ndarray):
                features = features.astype(np.float32)
                # Check if we need to transpose (frequency_bands, time_frames) -> (time_frames, frequency_bands)
                if features.shape[0] == self.cfg.n_bands and features.shape[1] != self.cfg.n_bands:
                    features = features.T  # Transpose to (time_frames, frequency_bands)
                # Cache the loaded features for faster access next time
                np.save(cp, features)
                return features
            else:
                raise ValueError(f"Pickle file {pickle_path} does not contain numpy array")
                
        except Exception as e:
            print(f"Error loading pickle {pickle_path}: {e}")
            # Fallback to dummy features if pickle loading fails
            dummy_features = np.random.normal(0, 0.1, (50, self.cfg.n_bands)).astype(np.float32)
            return dummy_features
    def create_silence_features(self, duration_ms: int) -> np.ndarray:
        n_frames = int(duration_ms * self.cfg.frame_hz / 1000)
        silence_features = np.random.normal(0, 0.01, (n_frames, self.cfg.n_bands)).astype(np.float32)
        return silence_features

def add_noise_variable_snr(features: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    ps = (features ** 2).mean(axis=0)
    noise_var = ps / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(size=features.shape).astype(np.float32) * np.sqrt(noise_var + 1e-12)
    return features + noise

# --- Speaker Split Lexicon ---
class SpeakerSplitLexicon:
    def __init__(self, data_dir: str, vocab_size: int, seed: int):
        # List of 15 specific speaker folder names
        speaker_names = [
            "Agnes", "Allison", "Bruce", "Junior", "Princess", "Samantha", "Tom", "Victoria",
            "Alex", "Ava", "Fred", "Kathy", "Ralph", "Susan", "Vicki"
        ]
        self.speaker_dirs = [os.path.join(data_dir, d) for d in speaker_names]
        self.speakers = speaker_names
        # Find intersection of words across all speakers
        speaker_words = []
        for speaker, d in zip(self.speakers, self.speaker_dirs):
            # Look for pickle files with naming convention: WORD_SPEAKER.pickle
            pickles = [f for f in os.listdir(d) if f.lower().endswith('.pickle')]
            words = []
            for f in pickles:
                # Extract word from filename like "ABSTAIN_AGNES.pickle"
                base_name = os.path.splitext(f)[0]  # Remove .pickle
                if '_' in base_name:
                    word_part, speaker_part = base_name.rsplit('_', 1)
                    if speaker_part.upper() == speaker.upper():
                        words.append(word_part.lower())  # Convert to lowercase for consistency
            speaker_words.append(set(words))
        shared_words = set.intersection(*speaker_words)
        if len(shared_words) < vocab_size:
            self.vocab = sorted(list(shared_words))
        else:
            random.seed(seed)
            self.vocab = sorted(random.sample(list(shared_words), vocab_size))
        # Dynamic test word allocation: vocab_size / #speakers words per speaker
        test_words_per_speaker = vocab_size // len(self.speakers)
        remaining_words = vocab_size % len(self.speakers)
        
        # Ensure we have enough words for the allocation
        total_test_words_needed = vocab_size
        if len(self.vocab) < total_test_words_needed:
            raise ValueError(f"Insufficient vocabulary for test split: need {total_test_words_needed}, have {len(self.vocab)}")
        
        # Allocate test words sequentially to ensure no repetition
        vocab_list = list(self.vocab)
        random.seed(seed)
        random.shuffle(vocab_list)  # Shuffle for randomness but then allocate deterministically
        
        self.test_words_by_speaker = {}
        word_idx = 0
        
        for i, speaker in enumerate(self.speakers):
            # First speakers get extra word if vocab_size doesn't divide evenly
            words_for_this_speaker = test_words_per_speaker + (1 if i < remaining_words else 0)
            
            speaker_test_words = vocab_list[word_idx:word_idx + words_for_this_speaker]
            self.test_words_by_speaker[speaker] = speaker_test_words
            word_idx += words_for_this_speaker
        # For each speaker, train words = vocab - their test word
        self.train_words_by_speaker = {s: [w for w in self.vocab if w not in self.test_words_by_speaker[s]]
                                       for s in self.speakers}
        # Map word to index
        self.words = self.vocab + ["<SILENCE>"]
        self.silence_idx = len(self.words) - 1
        self.stoi = {w: i for i, w in enumerate(self.words)}
        self.itos = {i: w for w, i in self.stoi.items()}
        
        # Verify no word repetition across speakers
        all_test_words = []
        for test_words in self.test_words_by_speaker.values():
            all_test_words.extend(test_words)
        if len(all_test_words) != len(set(all_test_words)):
            raise ValueError("Error: Test word repetition detected across speakers!")

# ...existing code for FeatureCache, add_noise_variable_snr, and dataset class, but modify dataset to accept speaker and split...

# --- Dataset for Speaker Split ---
class SpeakerWordDataset:
    def __init__(self, features_by_word: Dict[str, Dict[str, np.ndarray]], word_list: List[str],
                 speakers: list, split: str, test_words_by_speaker: Dict[str, list],
                 batch_size: int, seed: int, add_noise: bool = False, snr_db: float = 15.0,
                 silence_duration_ms: int = 200, silence_idx: int = -1, cache=None):
        self.features_by_word = features_by_word
        self.word_list = word_list
        self.speakers = speakers
        self.split = split
        self.batch_size = batch_size
        self.add_noise = add_noise
        self.rng = np.random.default_rng(seed)
        self.snr_db = snr_db
        self.silence_duration_ms = silence_duration_ms
        self.silence_idx = silence_idx
        self.cache = cache
        self.test_words_by_speaker = test_words_by_speaker
        # Build a list of (speaker, word) pairs for this split
        self.active_pairs = []
        for speaker in self.speakers:
            if split == 'train':
                words = [w for w in word_list if w not in test_words_by_speaker[speaker]]
            else:
                words = test_words_by_speaker[speaker]
            self.active_pairs.extend([(speaker, w) for w in words])
    def create_word_sequence(self, pairs: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        sequence_features = []
        frame_labels = []
        for i, (speaker, word) in enumerate(pairs):
            word_features = self.features_by_word[speaker][word].copy()
            if self.add_noise:
                word_features = add_noise_variable_snr(word_features, self.rng, self.snr_db)
            sequence_features.append(word_features)
            word_id = [j for j, w in enumerate(self.word_list) if w == word][0]
            word_labels = np.full(word_features.shape[0], word_id, dtype=np.int64)
            frame_labels.append(word_labels)
            if i < len(pairs) - 1:
                silence_features = self.cache.create_silence_features(self.silence_duration_ms)
                if self.add_noise:
                    silence_features = add_noise_variable_snr(silence_features, self.rng, self.snr_db)
                sequence_features.append(silence_features)
                silence_labels = np.full(silence_features.shape[0], self.silence_idx, dtype=np.int64)
                frame_labels.append(silence_labels)
        full_sequence = np.vstack(sequence_features)
        full_labels = np.concatenate(frame_labels)
        return full_sequence, full_labels
    def next_batch(self):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        batch_features = []
        batch_labels = []
        batch_word_sequences = []
        for _ in range(self.batch_size):
            pairs = [self.active_pairs[self.rng.integers(len(self.active_pairs))] for _ in range(3)]
            sequence_features, sequence_labels = self.create_word_sequence(pairs)
            batch_features.append(torch.from_numpy(sequence_features))
            batch_labels.append(torch.from_numpy(sequence_labels))
            batch_word_sequences.append(pairs)
        # Pad sequences to the same length
        batch_features_padded = pad_sequence(batch_features, batch_first=True)  # [B, T, D]
        batch_labels_padded = pad_sequence(batch_labels, batch_first=True, padding_value=-100)  # [B, T]
        return batch_features_padded, batch_labels_padded, batch_word_sequences


class CTM_ASR_Model(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.ctm = ContinuousThoughtMachineRL(
            iterations=cfg.iterations,
            d_model=cfg.d_model,
            d_input=cfg.d_input,
            n_synch_out=cfg.n_synch_out,
            synapse_depth=cfg.synapse_depth,
            memory_length=cfg.memory_length,
            deep_nlms=True,
            memory_hidden_dims=2,
            do_layernorm_nlm=False,
            backbone_type='classic-control-backbone',
            prediction_reshaper=[-1],
            dropout=cfg.dropout,
            neuron_select_type='first-last',
        )
        sync_dim = self.ctm.synch_representation_size_out
        self.head = nn.Linear(sync_dim, vocab_size)
    def initial_state(self, batch_size: int, device: torch.device):
        state_trace = torch.repeat_interleave(
            self.ctm.start_trace.unsqueeze(0), batch_size, 0
        ).to(device)
        activated_state_trace = torch.repeat_interleave(
            self.ctm.start_activated_trace.unsqueeze(0), batch_size, 0
        ).to(device)
        return state_trace, activated_state_trace
    def forward(self, batch_features, device: torch.device):
        # batch_features: [B, T, D]
        B, T, D = batch_features.shape
        batch_features = batch_features.to(device)
        
        # Initialize state for the whole batch
        st, ast = self.initial_state(B, device)
        
        # Process sequences more efficiently
        # Instead of looping through time, we can process in chunks for better memory usage
        chunk_size = min(32, T)  # Process in chunks to manage memory
        logits = []
        
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_features = batch_features[:, chunk_start:chunk_end, :]  # [B, chunk_size, D]
            
            chunk_logits = []
            for t in range(chunk_features.size(1)):
                x_t = chunk_features[:, t, :]  # [B, D] - No need for unsqueeze/squeeze
                sync_out, (st, ast) = self.ctm.forward(x_t, (st, ast))
                logits_t = self.head(sync_out)  # [B, vocab]
                chunk_logits.append(logits_t)
            
            # Stack chunk logits and append to main list
            if chunk_logits:
                chunk_logits_tensor = torch.stack(chunk_logits, dim=1)  # [B, chunk_size, vocab]
                logits.append(chunk_logits_tensor)
        
        # Concatenate all chunks
        if logits:
            logits = torch.cat(logits, dim=1)  # [B, T, vocab]
        else:
            logits = torch.zeros(B, T, len(self.head.weight), device=device)
            
        return logits

def validate_model(model, dataset: SpeakerWordDataset, device: torch.device, silence_idx: int) -> float:
    model.eval()
    correct = 0
    total = 0
    n_test_sequences = 5  # Reduce validation size for better practice
    with torch.no_grad():
        for _ in range(n_test_sequences):
            batch_features, batch_labels, _ = dataset.next_batch()
            logits = model(batch_features, device)  # [B, T, vocab]
            batch_labels = batch_labels.to(device)  # [B, T]
            # Mask out silence and padding
            for b in range(batch_labels.size(0)):
                labels = batch_labels[b]
                logit_seq = logits[b]
                word_mask = (labels != silence_idx) & (labels != -100)
                word_indices = torch.where(word_mask)[0]
                if len(word_indices) > 0:
                    # Use last 20 frames for validation (more stable than all frames)
                    eval_frames = min(20, len(word_indices))
                    last_indices = word_indices[-eval_frames:]
                    last_word_logits = logit_seq[last_indices]
                    last_word_labels = labels[last_indices]
                    preds = torch.argmax(last_word_logits, dim=1)
                    correct += (preds == last_word_labels).sum().item()
                    total += eval_frames
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return accuracy

# --- Main Training Loop for Speaker Split ---
def train_single_config(cfg: Config, experiment_name: str):
    """Train a single configuration and return results"""
    set_seed(cfg.seed)
    
    print(f"Setting up experiment {experiment_name}...")
    print(f"  - Data directory: {cfg.data_dir}")
    print(f"  - Config: iter={cfg.iterations}, vocab={cfg.vocab_size}, mem={cfg.memory_length}, d_model={cfg.d_model}")
    
    # Update cache and output directories for this experiment
    cfg.cache_dir = f"audio_cache/speaker_split_gammatone_64_100hz_vocab{cfg.vocab_size}"
    cfg.output_dir = f"outputs/ctm_gammatone_asr_experiments/{experiment_name}"
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(cfg.data_dir):
        print(f"ERROR: Data directory does not exist: {cfg.data_dir}")
        return None
    
    print(f"Initializing lexicon...")
    try:
        lex = SpeakerSplitLexicon(cfg.data_dir, cfg.vocab_size, cfg.seed)
        print(f"  - Found {len(lex.vocab)} words shared across {len(lex.speakers)} speakers")
    except Exception as e:
        print(f"ERROR: Failed to initialize lexicon: {e}")
        import traceback
        print(f"Lexicon error traceback:\n{traceback.format_exc()}")
        return None
    
    # Load features for all speakers/words with memory management
    print(f"Loading features for {len(lex.speakers)} speakers and {len(lex.vocab)} words...")
    cache = FeatureCache(cfg, lex)
    features_by_word = {}
    
    missing_files = []
    loaded_files = 0
    
    for speaker, speaker_dir in zip(lex.speakers, lex.speaker_dirs):
        if not os.path.exists(speaker_dir):
            print(f"WARNING: Speaker directory missing: {speaker_dir}")
            missing_files.append(f"Missing directory: {speaker_dir}")
            continue
            
        features_by_word[speaker] = {}
        for word in lex.vocab:
            # Use the new naming convention: WORD_SPEAKER.pickle
            pickle_file = f"{word.upper()}_{speaker.upper()}.pickle"
            pickle_path = os.path.join(speaker_dir, pickle_file)
            try:
                if os.path.exists(pickle_path):
                    features_by_word[speaker][word] = cache.get(f"{word}_{speaker}", pickle_path)
                    loaded_files += 1
                else:
                    missing_files.append(f"{pickle_path}")
                    # Create dummy features if file missing
                    features_by_word[speaker][word] = np.random.normal(0, 0.1, (50, cfg.n_bands)).astype(np.float32)
            except Exception as e:
                print(f"ERROR loading {pickle_path}: {e}")
                # Create dummy features if file missing (silently)
                features_by_word[speaker][word] = np.random.normal(0, 0.1, (50, cfg.n_bands)).astype(np.float32)
    
    print(f"  - Loaded {loaded_files} pickle files successfully")
    if missing_files:
        print(f"  - Missing {len(missing_files)} files (using dummy features)")
        if len(missing_files) <= 5:  # Show first few missing files
            for f in missing_files[:5]:
                print(f"    Missing: {f}")
    
    if loaded_files == 0:
        print(f"ERROR: No valid pickle files found! Check data directory and naming convention.")
        return None
    
    print(f"Training {experiment_name}: iter={cfg.iterations}, vocab={cfg.vocab_size}, mem={cfg.memory_length}, d_model={cfg.d_model}")
    
    print("Creating datasets...")
    try:
        train_dataset = SpeakerWordDataset(features_by_word, lex.vocab, lex.speakers, 'train',
                                           lex.test_words_by_speaker, cfg.batch_size, cfg.seed,
                                           add_noise=cfg.add_noise_train, snr_db=cfg.snr_db,
                                           silence_duration_ms=cfg.silence_duration_ms,
                                           silence_idx=lex.silence_idx, cache=cache)
        test_dataset = SpeakerWordDataset(features_by_word, lex.vocab, lex.speakers, 'test',
                                          lex.test_words_by_speaker, cfg.batch_size, cfg.seed+1,
                                          add_noise=False, snr_db=cfg.snr_db,
                                          silence_duration_ms=cfg.silence_duration_ms,
                                          silence_idx=lex.silence_idx, cache=cache)
        print(f"  - Train dataset: {len(train_dataset.active_pairs)} speaker-word pairs")
        print(f"  - Test dataset: {len(test_dataset.active_pairs)} speaker-word pairs")
    except Exception as e:
        print(f"ERROR: Failed to create datasets: {e}")
        import traceback
        print(f"Dataset error traceback:\n{traceback.format_exc()}")
        return None
    
    device = torch.device(cfg.device)
    
    # Initialize model with error handling
    print("Initializing model...")
    try:
        model = CTM_ASR_Model(cfg, vocab_size=len(lex.words)).to(device)
        # Initialize lazy parameters by doing a forward pass
        dummy_batch = torch.randn(1, 10, cfg.d_input).to(device)
        _ = model(dummy_batch, device)
        print(f"  - Model initialized successfully on {device}")
        print(f"  - Vocab size: {len(lex.words)} (including silence)")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"ERROR: Model initialization failed: {e}")
        import traceback
        print(f"Model error traceback:\n{traceback.format_exc()}")
        return None
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Add learning rate warmup
    def get_lr_scale(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        return 1.0
    
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "train_acc": [], "val_acc": []}
    
    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    global_step = 0
    total_steps = cfg.epochs * cfg.words_per_epoch
    pbar = tqdm(total=total_steps, desc=f"Training {experiment_name}", dynamic_ncols=True, leave=False)
    
    early_stopped = False
    start_time = time.time()
    
    for epoch in range(cfg.epochs):
        if early_stopped:
            break
            
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx in range(cfg.words_per_epoch):
            if early_stopped:
                break
                
            try:
                batch_features, batch_labels, batch_word_sequences = train_dataset.next_batch()
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass with error handling
                logits = model(batch_features, device)
                
                # Compute loss and accuracy
                loss = 0.0
                total_correct = 0
                total_frames = 0
                for b in range(batch_labels.size(0)):
                    labels = batch_labels[b]
                    logit_seq = logits[b]
                    word_mask = (labels != lex.silence_idx) & (labels != -100)
                    word_indices = torch.where(word_mask)[0]
                    if len(word_indices) > 0:
                        word_logits = logit_seq[word_indices]
                        word_labels = labels[word_indices]
                        sequence_loss = criterion(word_logits, word_labels)
                        loss += sequence_loss
                        preds = torch.argmax(word_logits, dim=1)
                        total_correct += (preds == word_labels).sum().item()
                        total_frames += len(word_indices)
                        
                loss = loss / batch_labels.size(0) if batch_labels.size(0) > 0 else torch.tensor(0.0)
                train_acc = total_correct / total_frames if total_frames > 0 else 0.0
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                
                # Apply learning rate warmup
                lr_scale = get_lr_scale(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * lr_scale
                
                optimizer.step()
                
                # Track metrics
                history["loss"].append(float(loss.item()))
                history["train_acc"].append(train_acc)
                epoch_loss += loss.item()
                epoch_correct += total_correct
                epoch_total += total_frames
                
                # Validation
                if global_step % cfg.validate_every == 0:
                    val_acc = validate_model(model, test_dataset, device, lex.silence_idx)
                    history["val_acc"].append(val_acc)
                    scheduler.step(val_acc)
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Early stopping check
                    if val_acc > best_val_acc + cfg.min_improvement:
                        best_val_acc = val_acc
                        patience_counter = 0
                        best_model_state = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': val_acc,
                            'epoch': epoch,
                            'global_step': global_step
                        }
                        status = "NEW BEST"
                    else:
                        patience_counter += 1
                        status = f"({patience_counter}/{cfg.early_stop_patience})"
                    
                    pbar.set_postfix({
                        "loss": f"{loss.item():.3f}", 
                        "train_acc": f"{train_acc:.3f}", 
                        "val_acc": f"{val_acc:.3f}",
                        "best": f"{best_val_acc:.3f}",
                        "lr": f"{current_lr:.1e}",
                        "status": status
                    })
                    
                    # Check if we should stop early
                    if patience_counter >= cfg.early_stop_patience:
                        early_stopped = True
                        break
                        
            except Exception as e:
                continue
                        
            global_step += 1
            pbar.update(1)
    
    pbar.close()
    training_time = time.time() - start_time
    
    # Restore best model if we have one
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    # Add experiment info to history
    history["experiment_info"] = {
        "experiment_name": experiment_name,
        "vocab_size": cfg.vocab_size,
        "iterations": cfg.iterations,
        "memory_length": cfg.memory_length,
        "d_model": cfg.d_model,
        "early_stopped": early_stopped,
        "best_val_acc": best_val_acc,
        "final_epoch": epoch,
        "final_step": global_step,
        "training_time_seconds": training_time,
        "device": str(device),
        "speakers": lex.speakers,
        "vocab": lex.vocab,
        "test_words_by_speaker": lex.test_words_by_speaker
    }
    
    # Save individual results
    with open(os.path.join(cfg.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Completed {experiment_name}: Best Val Acc = {best_val_acc:.4f}")
    return history

def run_comparison_experiments():
    """Run multiple experiments with different configurations"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    base_cfg = Config()
    
    # Define experiment configurations - optimized for comparison
    experiments = [
        # Varying iterations (ticks) - most important comparison
        {"name": "iter1_vocab15_mem10", "iterations": 1, "vocab_size": 15, "memory_length": 10},
        {"name": "iter2_vocab15_mem10", "iterations": 2, "vocab_size": 15, "memory_length": 10},
        {"name": "iter3_vocab15_mem10", "iterations": 3, "vocab_size": 15, "memory_length": 10},
        {"name": "iter4_vocab15_mem10", "iterations": 4, "vocab_size": 15, "memory_length": 10},
        
        # Varying iterations with vocab 50
        {"name": "iter1_vocab50_mem10", "iterations": 1, "vocab_size": 50, "memory_length": 10},
        {"name": "iter2_vocab50_mem10", "iterations": 2, "vocab_size": 50, "memory_length": 10},
        {"name": "iter3_vocab50_mem10", "iterations": 3, "vocab_size": 50, "memory_length": 10},
        {"name": "iter4_vocab50_mem10", "iterations": 4, "vocab_size": 50, "memory_length": 10},
        
        # Varying memory length - extended range for better word coverage
        {"name": "iter2_vocab15_mem5", "iterations": 2, "vocab_size": 15, "memory_length": 5},
        {"name": "iter2_vocab15_mem15", "iterations": 2, "vocab_size": 15, "memory_length": 15},
        {"name": "iter2_vocab15_mem20", "iterations": 2, "vocab_size": 15, "memory_length": 20},
        {"name": "iter2_vocab15_mem40", "iterations": 2, "vocab_size": 15, "memory_length": 40},
        {"name": "iter2_vocab15_mem60", "iterations": 2, "vocab_size": 15, "memory_length": 60},
        {"name": "iter2_vocab15_mem80", "iterations": 2, "vocab_size": 15, "memory_length": 80},
        {"name": "iter2_vocab15_mem120", "iterations": 2, "vocab_size": 15, "memory_length": 120},
        
        # Varying memory length with vocab 50 - extended range
        {"name": "iter2_vocab50_mem5", "iterations": 2, "vocab_size": 50, "memory_length": 5},
        {"name": "iter2_vocab50_mem15", "iterations": 2, "vocab_size": 50, "memory_length": 15},
        {"name": "iter2_vocab50_mem20", "iterations": 2, "vocab_size": 50, "memory_length": 20},
        {"name": "iter2_vocab50_mem40", "iterations": 2, "vocab_size": 50, "memory_length": 40},
        {"name": "iter2_vocab50_mem60", "iterations": 2, "vocab_size": 50, "memory_length": 60},
        {"name": "iter2_vocab50_mem80", "iterations": 2, "vocab_size": 50, "memory_length": 80},
        {"name": "iter2_vocab50_mem120", "iterations": 2, "vocab_size": 50, "memory_length": 120},
        
        # Varying vocabulary size
        {"name": "iter2_vocab15_mem10", "iterations": 2, "vocab_size": 15, "memory_length": 10},
        {"name": "iter2_vocab50_mem10", "iterations": 2, "vocab_size": 50, "memory_length": 10},
        
        # Varying model size
        {"name": "iter2_vocab15_mem10_d128", "iterations": 2, "vocab_size": 15, "memory_length": 10, "d_model": 128},
        {"name": "iter2_vocab15_mem10_d384", "iterations": 2, "vocab_size": 15, "memory_length": 10, "d_model": 384},
        
        # Varying model size with vocab 50
        {"name": "iter2_vocab50_mem10_d128", "iterations": 2, "vocab_size": 50, "memory_length": 10, "d_model": 128},
        {"name": "iter2_vocab50_mem10_d384", "iterations": 2, "vocab_size": 50, "memory_length": 10, "d_model": 384},
    ]
    
    all_results = {}
    
    # Check if we should run in parallel or sequential
    if torch.cuda.is_available() and base_cfg.max_parallel_experiments > 1:
        print("Running experiments sequentially (GPU mode)")
        # Run sequentially when using GPU to avoid memory issues
        for exp_config in experiments:
            cfg = Config()
            for key, value in exp_config.items():
                if key != "name":
                    setattr(cfg, key, value)
            
            try:
                print(f"Starting experiment: {exp_config['name']}")
                history = train_single_config(cfg, exp_config["name"])
                if history is not None:
                    all_results[exp_config["name"]] = history
                    print(f"✓ Successfully completed: {exp_config['name']}")
                else:
                    print(f"✗ Failed (returned None): {exp_config['name']}")
                
                # Clear GPU cache between experiments
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"✗ Error in experiment {exp_config['name']}: {str(e)}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                continue
    else:
        print(f"Running experiments in parallel (CPU mode)")
        # Run in parallel for CPU
        def run_single_experiment(exp_config):
            cfg = Config()
            cfg.device = "cpu"  # Force CPU for parallel execution
            for key, value in exp_config.items():
                if key != "name":
                    setattr(cfg, key, value)
            
            try:
                return exp_config["name"], train_single_config(cfg, exp_config["name"])
            except Exception as e:
                return exp_config["name"], None
        
        with ProcessPoolExecutor(max_workers=base_cfg.max_parallel_experiments) as executor:
            future_to_exp = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
            
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    name, result = future.result()
                    if result is not None:
                        all_results[name] = result
                except Exception as e:
                    continue
    
    # Generate comparison plots
    if all_results:
        generate_comparison_plots(all_results)
        print(f"Completed {len(all_results)}/{len(experiments)} experiments successfully!")
        print("Results saved to outputs/ctm_gammatone_asr_experiments/")
    else:
        print("No experiments completed successfully")
    
    return all_results

def generate_comparison_plots(all_results):
    """Generate comprehensive comparison plots"""
    if not all_results:
        print("No results to plot")
        return
    
    # Filter out None results and report
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Create output directory for comparison plots
    comparison_dir = "outputs/ctm_gammatone_asr_experiments/plots"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Iterations Comparison
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Iterations vs Final Performance
    plt.subplot(3, 4, 1)
    iter_results = {k: v for k, v in valid_results.items() 
                   if "vocab15_mem10" in k and "iter" in k and "d128" not in k and "d384" not in k}
    iterations = []
    final_accs = []
    for name, result in iter_results.items():
        if 'experiment_info' in result:
            iter_num = int(name.split("iter")[1].split("_")[0])
            iterations.append(iter_num)
            final_accs.append(result['experiment_info']['best_val_acc'])
    
    if iterations:
        plt.scatter(iterations, final_accs, s=100, alpha=0.7)
        plt.plot(iterations, final_accs, 'b--', alpha=0.5)
        for i, (x, y) in enumerate(zip(iterations, final_accs)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Number of Iterations (Ticks)')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Performance vs CTM Iterations')
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Memory Length vs Performance
    plt.subplot(3, 4, 2)
    mem_results = {k: v for k, v in valid_results.items() 
                  if "iter2_vocab15" in k and "mem" in k and "d128" not in k and "d384" not in k}
    memory_lengths = []
    mem_final_accs = []
    for name, result in mem_results.items():
        if 'experiment_info' in result:
            if "mem5" in name:
                mem_len = 5
            elif "mem15" in name:
                mem_len = 15
            elif "mem20" in name:
                mem_len = 20
            else:
                mem_len = 10
            memory_lengths.append(mem_len)
            mem_final_accs.append(result['experiment_info']['best_val_acc'])
    
    if memory_lengths:
        plt.scatter(memory_lengths, mem_final_accs, s=100, alpha=0.7, color='green')
        plt.plot(memory_lengths, mem_final_accs, 'g--', alpha=0.5)
        for i, (x, y) in enumerate(zip(memory_lengths, mem_final_accs)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Memory Length')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Performance vs Memory Length')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Vocabulary Size vs Performance
    plt.subplot(3, 4, 3)
    vocab_results = {k: v for k, v in valid_results.items() 
                    if "iter2" in k and "mem10" in k and "d128" not in k and "d384" not in k}
    vocab_sizes = []
    vocab_final_accs = []
    for name, result in vocab_results.items():
        if 'experiment_info' in result:
            vocab_size = result['experiment_info']['vocab_size']
            vocab_sizes.append(vocab_size)
            vocab_final_accs.append(result['experiment_info']['best_val_acc'])
    
    if vocab_sizes:
        plt.scatter(vocab_sizes, vocab_final_accs, s=100, alpha=0.7, color='red')
        plt.plot(vocab_sizes, vocab_final_accs, 'r--', alpha=0.5)
        for i, (x, y) in enumerate(zip(vocab_sizes, vocab_final_accs)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Vocabulary Size')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Performance vs Vocabulary Size')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Model Size vs Performance
    plt.subplot(3, 4, 4)
    model_results = {k: v for k, v in valid_results.items() 
                    if "iter2_vocab15_mem10" in k}
    model_sizes = []
    model_final_accs = []
    for name, result in model_results.items():
        if 'experiment_info' in result:
            d_model = result['experiment_info']['d_model']
            model_sizes.append(d_model)
            model_final_accs.append(result['experiment_info']['best_val_acc'])
    
    if model_sizes:
        plt.scatter(model_sizes, model_final_accs, s=100, alpha=0.7, color='purple')
        plt.plot(model_sizes, model_final_accs, 'purple', linestyle='--', alpha=0.5)
        for i, (x, y) in enumerate(zip(model_sizes, model_final_accs)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Model Dimension (d_model)')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Performance vs Model Size')
        plt.grid(True, alpha=0.3)
    
    # Plot 5-8: Training curves for different iterations
    colors = ['blue', 'green', 'red', 'purple']
    valid_iter_results = [(name, result) for name, result in iter_results.items() 
                         if 'val_acc' in result and result['val_acc']]
    for i, (name, result) in enumerate(valid_iter_results[:4]):
        plt.subplot(3, 4, 5 + i)
        if result['val_acc']:
            val_steps = [j * 20 for j in range(len(result['val_acc']))]  # validate_every = 20
            plt.plot(val_steps, result['val_acc'], color=colors[i], linewidth=2)
            plt.title(f'{name}: {result["experiment_info"]["iterations"]} iterations')
            plt.xlabel('Training Step')
            plt.ylabel('Validation Accuracy')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
    
    # Plot 9-12: Training curves for different memory lengths
    valid_mem_results = [(k, v) for k, v in mem_results.items() 
                        if 'val_acc' in v and v['val_acc']]
    for i, (name, result) in enumerate(valid_mem_results[:4]):
        plt.subplot(3, 4, 9 + i)
        if result['val_acc']:
            val_steps = [j * 20 for j in range(len(result['val_acc']))]
            plt.plot(val_steps, result['val_acc'], color=colors[i], linewidth=2)
            plt.title(f'{name}: mem={result["experiment_info"]["memory_length"]}')
            plt.xlabel('Training Step')
            plt.ylabel('Validation Accuracy')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'ctm_parameter_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate summary table
    generate_summary_table(valid_results, comparison_dir)

def generate_summary_table(all_results, output_dir):
    """Generate a summary table of all experiments"""
    summary_data = []
    
    for name, result in all_results.items():
        exp_info = result['experiment_info']
        summary_data.append({
            'Experiment': name,
            'Iterations': exp_info['iterations'],
            'Vocab Size': exp_info['vocab_size'],
            'Memory Length': exp_info['memory_length'],
            'Model Dim': exp_info['d_model'],
            'Best Val Acc': f"{exp_info['best_val_acc']:.4f}",
            'Final Epoch': exp_info['final_epoch'],
            'Early Stopped': exp_info['early_stopped']
        })
    
    # Sort by best validation accuracy
    summary_data.sort(key=lambda x: float(x['Best Val Acc']), reverse=True)
    
    # Save as JSON
    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed summary as text
    with open(os.path.join(output_dir, 'experiment_summary.txt'), 'w') as f:
        f.write("EXPERIMENT SUMMARY (sorted by best validation accuracy)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Experiment':<25} {'Iter':<4} {'Vocab':<5} {'Mem':<4} {'Dim':<4} {'Best Acc':<8} {'Epoch':<5} {'Early Stop'}\n")
        f.write("-"*80 + "\n")
        
        for row in summary_data:
            f.write(f"{row['Experiment']:<25} {row['Iterations']:<4} {row['Vocab Size']:<5} {row['Memory Length']:<4} {row['Model Dim']:<4} {row['Best Val Acc']:<8} {row['Final Epoch']:<5} {row['Early Stopped']}\n")
        
        f.write("="*80 + "\n")

# Modified main training function (now calls comparison experiments)
def train(cfg: Config):
    """Legacy function for backward compatibility - now runs comparison experiments"""
    return run_comparison_experiments()

if __name__ == "__main__":
    # Run all comparison experiments
    print("Starting CTM ASR comparison experiments...")
    
    # First, let's check if the data directory exists and list its contents
    data_dir = "GAMMATONE_64_100"
    print(f"\nChecking data directory: {data_dir}")
    if os.path.exists(data_dir):
        print(f"✓ Data directory exists")
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"  Found {len(subdirs)} subdirectories: {subdirs}")
        
        # Check a few speaker directories for pickle files
        sample_speakers = subdirs[:3] if subdirs else []
        for speaker in sample_speakers:
            speaker_path = os.path.join(data_dir, speaker)
            pickles = [f for f in os.listdir(speaker_path) if f.endswith('.pickle')]
            print(f"  {speaker}: {len(pickles)} pickle files")
            if pickles:
                print(f"    Sample files: {pickles[:3]}")
    else:
        print(f"✗ Data directory does not exist: {data_dir}")
        print("Please check your data directory path.")
        exit(1)
    
    results = run_comparison_experiments()
    print(f"Experiments completed. Results in outputs/ctm_gammatone_asr_experiments/")
