"""
PyTorch-first Parakeet TDT model — no NeMo required at inference time.

Architecture: FastConformer encoder + TDT (Token-and-Duration Transducer) decoder.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TranscriptionResult:
    """Returned by transcribe(..., timestamps=True).

    Attributes:
        text: Full decoded transcript.
        timestamp: Dict with keys 'char', 'word', 'segment', each a list of
            dicts containing the text span and 'start'/'end' in seconds.

    Example::

        result = model.transcribe("audio.wav", timestamps=True)
        for w in result.timestamp['word']:
            print(f"{w['start']:.2f}s – {w['end']:.2f}s : {w['word']}")
    """
    text: str
    timestamp: dict  # {'char': [...], 'word': [...], 'segment': [...]}


class LogMelPreprocessor(nn.Module):
    """Log mel spectrogram + per-feature normalization (mirrors NeMo's featurizer)."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160,
                 win_length: int = 400, n_mels: int = 128,
                 log_zero_guard: float = 5.960464e-8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_zero_guard = log_zero_guard
        self.register_buffer('window', torch.zeros(win_length))
        self.register_buffer('fb', torch.zeros(1, n_mels, n_fft // 2 + 1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [T] float32, 16 kHz mono
        Returns:
            features: [1, n_mels, T'] normalised log-mel spectrogram
        """
        x = audio.unsqueeze(0)  # [1, T]
        x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)

        with torch.amp.autocast(x.device.type, enabled=False):
            stft = torch.stft(
                x.float(), n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=self.window.float(),
                center=True, pad_mode='reflect', return_complex=True,
            )  # [1, n_fft//2+1, T']

        power = torch.view_as_real(stft).pow(2).sum(-1)  # [1, 257, T']
        mel     = torch.matmul(self.fb.to(power.dtype), power)  # [1, 128, T']
        log_mel = torch.log(mel + self.log_zero_guard)

        T    = log_mel.shape[-1]
        mean = log_mel.sum(dim=-1, keepdim=True) / T
        std  = torch.sqrt(((log_mel - mean).pow(2).sum(dim=-1, keepdim=True)) / (T - 1))
        return (log_mel - mean) / (std + 1e-5)  # [1, n_mels, T']


class ConvSubsampling(nn.Module):
    """3-layer stride-2 Conv2D subsampling (8× downsampling) + linear projection.

    Input shape expected: [B, T, n_mels] (NeMo transposes before calling this).
    Output shape:         [B, T//8, d_model]
    """

    def __init__(self, d_model: int = 1024, n_mels: int = 128):
        super().__init__()
        freq_out = n_mels // 8  # = 16 after three stride-2 ops
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256),  # depthwise
            nn.Conv2d(256, 256, 1),                                     # pointwise
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256),  # depthwise
            nn.Conv2d(256, 256, 1),                                     # pointwise
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(256 * freq_out, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x:       [B, T, n_mels]
        lengths: [B] (time lengths in frames)
        Returns: out [B, T', d_model], new_lengths [B]
        """
        x = x.unsqueeze(1)  # [B, 1, T, n_mels]
        x = self.conv(x)    # [B, 256, T//8, n_mels//8]
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))  # [B, T', d_model]
        new_lengths = lengths
        for _ in range(3):
            new_lengths = torch.div(new_lengths - 1, 2, rounding_mode='floor') + 1
        return x, new_lengths.to(torch.long)


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding (Transformer-XL style) used by FastConformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self._build_pe(max_len)

    def _build_pe(self, length: int, device=None, dtype=None):
        needed = 2 * length - 1
        pos = torch.arange(length - 1, -length, -1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(needed, self.d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, 2L-1, d_model]
        if device is not None:
            pe = pe.to(device=device, dtype=dtype)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, d_model]
        Returns: x (unchanged), pos_emb [1, 2T-1, d_model]
        """
        T = x.size(1)
        needed = 2 * T - 1
        if not hasattr(self, 'pe') or self.pe.size(1) < needed:
            self._build_pe(T, device=x.device, dtype=x.dtype)
        center = self.pe.size(1) // 2 + 1
        pos_emb = self.pe[:, center - T: center + T - 1]
        return x, pos_emb


class RelPositionMHA(nn.Module):
    """Relative-position Multi-Head Attention (Transformer-XL / NeMo style)."""

    def __init__(self, d_model: int = 1024, n_heads: int = 8):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q   = nn.Linear(d_model, d_model, bias=False)
        self.linear_k   = nn.Linear(d_model, d_model, bias=False)
        self.linear_v   = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.shape
        x = F.pad(x, (1, 0))             # [B, h, T, 2T]
        x = x.view(b, h, -1, qlen)       # [B, h, 2T, T]
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # [B, h, T, 2T-1]
        return x

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:       [B, T, d_model]
        pos_emb: [1, 2T-1, d_model]
        mask:    [B, T, T] bool, True = ignore (padding)
        """
        B, T, _ = x.shape

        q = self.linear_q(x).view(B, T, self.h, self.d_k)
        k = self.linear_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)

        q_u = (q + self.pos_bias_u).transpose(1, 2)
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))

        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)[:, :, :, :T]

        scores = (matrix_ac + matrix_bd) / self.s_d_k

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).reshape(B, T, self.h * self.d_k)
        return self.linear_out(out)


class ConformerFeedForward(nn.Module):
    """Macaron-style feed-forward (no bias, Swish/SiLU activation)."""

    def __init__(self, d_model: int = 1024, d_ff: int = 4096):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class ConformerConvModule(nn.Module):
    """Conformer convolution sub-layer: PW → GLU → DW → BN → SiLU → PW."""

    def __init__(self, d_model: int = 1024, kernel_size: int = 9):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)
        self.depthwise_conv  = nn.Conv1d(d_model, d_model, kernel_size,
                                          padding=kernel_size // 2, groups=d_model,
                                          bias=False)
        self.batch_norm      = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.act             = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]"""
        x = x.transpose(1, 2)        # [B, d, T]
        x = self.pointwise_conv1(x)  # [B, 2d, T]
        x = F.glu(x, dim=1)          # [B, d, T]
        x = self.depthwise_conv(x)   # [B, d, T]
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)  # [B, d, T]
        return x.transpose(1, 2)     # [B, T, d]


class FastConformerLayer(nn.Module):
    """Single FastConformer block: FF1 → Self-Attn → Conv → FF2 → LayerNorm."""

    def __init__(self, d_model: int = 1024, n_heads: int = 8,
                 d_ff: int = 4096, conv_kernel: int = 9):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1      = ConformerFeedForward(d_model, d_ff)
        self.norm_conv          = nn.LayerNorm(d_model)
        self.conv               = ConformerConvModule(d_model, conv_kernel)
        self.norm_self_att      = nn.LayerNorm(d_model)
        self.self_attn          = RelPositionMHA(d_model, n_heads)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2      = ConformerFeedForward(d_model, d_ff)
        self.norm_out           = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # NeMo order: FF1 → Self-Attn → Conv → FF2 → LayerNorm
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class FastConformerEncoder(nn.Module):
    """24-layer FastConformer encoder with relative positional encoding."""

    def __init__(self, d_model: int = 1024, n_layers: int = 24,
                 n_heads: int = 8, d_ff: int = 4096,
                 n_mels: int = 128, conv_kernel: int = 9):
        super().__init__()
        self.pre_encode = ConvSubsampling(d_model, n_mels)
        self.pos_enc    = RelPositionalEncoding(d_model)
        self.layers     = nn.ModuleList([
            FastConformerLayer(d_model, n_heads, d_ff, conv_kernel)
            for _ in range(n_layers)
        ])

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        """
        features: [B, n_mels, T]
        lengths:  [B]
        Returns:  encoder_out [B, T', d_model],  new_lengths [B]
        """
        x = features.transpose(1, 2)
        x, lengths = self.pre_encode(x, lengths)
        x, pos_emb = self.pos_enc(x)

        B, T, _ = x.shape
        pad = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        mask = pad.unsqueeze(1).expand(B, T, T)
        mask = mask if pad.any() else None

        for layer in self.layers:
            x = layer(x, pos_emb, mask)
        return x, lengths


class RNNTDecoder(nn.Module):
    """RNNT prediction network: embedding + 2-layer LSTM (blank_as_pad=True)."""

    def __init__(self, vocab_size: int = 8192, d_model: int = 640, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model)
        self.lstm  = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)

    def forward(self, labels: torch.Tensor, hidden=None):
        """
        labels: [B, 1]  token indices (blank index for SOS)
        Returns: output [B, 1, d_model], new hidden state
        """
        x, hidden = self.lstm(self.embed(labels), hidden)
        return x, hidden


class TDTJoint(nn.Module):
    """TDT joint network: projects encoder + decoder outputs → token + duration logits."""

    def __init__(self, enc_dim: int = 1024, pred_dim: int = 640,
                 joint_dim: int = 640, vocab_size: int = 8192, n_durations: int = 5):
        super().__init__()
        self.enc  = nn.Linear(enc_dim, joint_dim)
        self.pred = nn.Linear(pred_dim, joint_dim)
        self.joint_net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),  # no-op in eval mode
            nn.Linear(joint_dim, vocab_size + 1 + n_durations),  # 8198
        )

    def forward(self, enc_out: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        """
        enc_out:  [B, 1, enc_dim]
        pred_out: [B, 1, pred_dim]
        Returns:  [B, 1, 1, vocab+1+n_dur]
        """
        return self.joint_net(self.enc(enc_out) + self.pred(pred_out)).unsqueeze(2)


def _frames_to_timestamps(
    token_ids: list,
    token_frames: list,
    sp,
    enc_len: int,
) -> dict:
    """Convert per-token encoder-frame ranges to char/word/segment timestamps.

    Args:
        token_ids:    List of token IDs from the decoder.
        token_frames: List of (start_frame, end_frame) parallel to token_ids.
        sp:           SentencePieceProcessor for piece→text mapping.
        enc_len:      Total encoder sequence length (used to close the last span).
    Returns:
        dict with keys 'char', 'word', 'segment'.
    """
    # 1 encoder frame = 8 mel frames × hop_length / sample_rate = 80 ms
    FRAME_S = 8 * 160 / 16_000  # 0.08 s

    # Fix zero-duration tokens (skip=0, label-looping): extend to next token's start.
    frames = list(token_frames)
    for i in range(len(frames) - 1):
        if frames[i][1] == frames[i][0]:
            frames[i] = (frames[i][0], frames[i + 1][0])
    if frames and frames[-1][1] == frames[-1][0]:
        frames[-1] = (frames[-1][0], frames[-1][0] + 1)

    char_ts  = []
    word_ts  = []
    curr_word_text   = ''
    curr_word_start  = None
    curr_word_end    = None
    first_token      = True
    SENT_END = {'.', '!', '?'}

    for tok_id, (start_f, end_f) in zip(token_ids, frames):
        piece        = sp.IdToPiece(tok_id)
        is_word_start = piece.startswith('\u2581')   # '▁'
        raw_text     = piece.lstrip('\u2581')

        start_s = start_f * FRAME_S
        end_s   = end_f   * FRAME_S

        # --- char timestamps ---
        # Prepend a space character for word-boundary tokens (except the first).
        all_chars = ([' '] if is_word_start and not first_token else []) + list(raw_text)
        n = max(len(all_chars), 1)
        dur = (end_s - start_s) / n
        for j, ch in enumerate(all_chars):
            char_ts.append({
                'char':  ch,
                'start': round(start_s + j * dur, 3),
                'end':   round(start_s + (j + 1) * dur, 3),
            })

        # --- word timestamps ---
        if is_word_start and not first_token:
            if curr_word_text:
                word_ts.append({
                    'word':  curr_word_text,
                    'start': round(curr_word_start, 3),
                    'end':   round(curr_word_end, 3),
                })
            curr_word_text  = raw_text
            curr_word_start = start_s
            curr_word_end   = end_s
        else:
            curr_word_text += raw_text
            if curr_word_start is None:
                curr_word_start = start_s
            curr_word_end = end_s

        first_token = False

    # Flush last word.
    if curr_word_text:
        word_ts.append({
            'word':  curr_word_text,
            'start': round(curr_word_start, 3),
            'end':   round(curr_word_end, 3),
        })

    # --- segment timestamps: split on sentence-ending punctuation ---
    seg_ts   = []
    curr_seg = []
    for w in word_ts:
        curr_seg.append(w)
        if w['word'] and w['word'][-1] in SENT_END:
            seg_ts.append({
                'segment': ' '.join(x['word'] for x in curr_seg),
                'start':   curr_seg[0]['start'],
                'end':     curr_seg[-1]['end'],
            })
            curr_seg = []
    if curr_seg:
        seg_ts.append({
            'segment': ' '.join(x['word'] for x in curr_seg),
            'start':   curr_seg[0]['start'],
            'end':     curr_seg[-1]['end'],
        })

    return {'char': char_ts, 'word': word_ts, 'segment': seg_ts}


@torch.inference_mode()
def tdt_greedy_decode(
    encoder_out: torch.Tensor,
    enc_len: int,
    decoder: RNNTDecoder,
    joint: TDTJoint,
    blank_id: int,
    durations: list,
    max_symbols_per_step: int = 10,
    return_timestamps: bool = False,
) -> list:
    """TDT greedy decoding for a single sequence (mirrors NeMo's GreedyTDTInfer).

    When return_timestamps=True returns (tokens, token_frames) where token_frames
    is a list of (start_frame, end_frame) pairs in encoder-frame units.
    """
    device = encoder_out.device
    tokens       = []
    token_frames = []  # (start_frame, end_frame) per token
    hidden       = None
    last_label   = torch.tensor([[blank_id]], dtype=torch.long, device=device)

    time_idx = 0
    skip     = 1
    while time_idx < enc_len:
        f = encoder_out[time_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]

        symbols_added = 0
        need_loop     = True

        while need_loop and symbols_added < max_symbols_per_step:
            pred_out, hidden_prime = decoder(last_label, hidden)
            logits = joint(f, pred_out)[0, 0, 0, :]  # [8198]

            n_dur         = len(durations)
            token_logits  = logits[:-n_dur]
            dur_logits    = logits[-n_dur:]

            k    = int(token_logits.argmax())
            d_k  = int(F.log_softmax(dur_logits, dim=-1).argmax())
            skip = durations[d_k]

            if k == blank_id:
                need_loop = False
            else:
                tokens.append(k)
                if return_timestamps:
                    token_frames.append((time_idx, time_idx + skip))
                hidden     = hidden_prime
                last_label = torch.tensor([[k]], dtype=torch.long, device=device)

            symbols_added += 1
            time_idx      += skip
            need_loop      = need_loop and (skip == 0)

        if skip == 0:
            skip = 1

        if symbols_added == max_symbols_per_step:
            time_idx += 1

    if return_timestamps:
        return tokens, token_frames
    return tokens


class ParakeetTDT(nn.Module):
    VOCAB_SIZE = 8192
    BLANK_ID   = 8192
    DURATIONS  = [0, 1, 2, 3, 4]

    def __init__(self):
        super().__init__()
        self.preprocessor  = LogMelPreprocessor()
        self.encoder       = FastConformerEncoder()
        self.decoder       = RNNTDecoder()
        self.joint         = TDTJoint()
        self.sp            = None  # SentencePieceProcessor, set by from_pretrained()
        self._decode_graph = None  # built lazily on first inference

    @torch.inference_mode()
    def warmup(self, duration_s: float = 5.0):
        """Run a dummy inference to trigger CUDA graph capture and warm up CUDA kernels."""
        device = next(self.parameters()).device
        dummy  = torch.zeros(int(16000 * duration_s), device=device)
        self.transcribe_audio(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def _build_decode_graph(self, device: torch.device):
        """Capture a CUDA graph for the per-step decoder+joint computation."""
        enc_dim  = self.joint.enc.in_features
        pred_dim = self.decoder.lstm.hidden_size

        dtype = next(self.parameters()).dtype
        self._g_label = torch.full((1, 1), self.BLANK_ID, dtype=torch.long, device=device)
        self._g_f     = torch.zeros(1, 1, enc_dim,  dtype=dtype, device=device)
        self._g_h     = torch.zeros(self.decoder.lstm.num_layers, 1, pred_dim, dtype=dtype, device=device)
        self._g_c     = torch.zeros(self.decoder.lstm.num_layers, 1, pred_dim, dtype=dtype, device=device)

        for _ in range(3):
            emb = self.decoder.embed(self._g_label)
            _pred, _ = self.decoder.lstm(emb, (self._g_h, self._g_c))
            self.joint(self._g_f, _pred)
        torch.cuda.synchronize()

        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph):
            emb = self.decoder.embed(self._g_label)
            pred, (self._g_h_out, self._g_c_out) = self.decoder.lstm(
                emb, (self._g_h, self._g_c)
            )
            self._g_logits = self.joint(self._g_f, pred)[0, 0, 0, :]

    @torch.inference_mode()
    def transcribe_audio(self, audio: torch.Tensor,
                         return_timestamps: bool = False):
        """
        audio: 1D float32 tensor, 16 kHz
        Returns: list of token IDs, or (token_ids, token_frames, enc_len) when
                 return_timestamps=True.
        """
        device = next(self.parameters()).device
        if device.type == 'cuda' and self._decode_graph is None:
            self._build_decode_graph(device)

        audio    = audio.to(device)
        features = self.preprocessor(audio)
        T        = features.shape[-1]
        lengths  = torch.tensor([T], device=device, dtype=torch.long)
        enc_dtype = next(self.encoder.parameters()).dtype
        if enc_dtype == torch.float32:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                enc_out, enc_lengths = self.encoder(features, lengths)
        else:
            enc_out, enc_lengths = self.encoder(features.to(enc_dtype), lengths)
        enc_len     = int(enc_lengths[0])
        encoder_out = enc_out[0].float()

        if self._decode_graph is not None:
            return self._tdt_decode_graphed(encoder_out, enc_len, return_timestamps)
        return tdt_greedy_decode(
            encoder_out=encoder_out, enc_len=enc_len,
            decoder=self.decoder, joint=self.joint,
            blank_id=self.BLANK_ID, durations=self.DURATIONS,
            return_timestamps=return_timestamps,
        )

    def _tdt_decode_graphed(self, encoder_out: torch.Tensor, enc_len: int,
                             return_timestamps: bool = False):
        """TDT greedy decode using the pre-captured CUDA graph for each step."""
        n_dur        = len(self.DURATIONS)
        tokens       = []
        token_frames = []

        self._g_h.zero_()
        self._g_c.zero_()
        self._g_label.fill_(self.BLANK_ID)

        time_idx = 0
        skip     = 1
        while time_idx < enc_len:
            self._g_f.copy_(encoder_out[time_idx].unsqueeze(0).unsqueeze(0))

            symbols_added = 0
            need_loop     = True
            while need_loop and symbols_added < 10:
                self._decode_graph.replay()

                k    = int(self._g_logits[:-n_dur].argmax())
                d_k  = int(F.log_softmax(self._g_logits[-n_dur:], dim=-1).argmax())
                skip = self.DURATIONS[d_k]

                if k == self.BLANK_ID:
                    need_loop = False
                else:
                    tokens.append(k)
                    if return_timestamps:
                        token_frames.append((time_idx, time_idx + skip))
                    self._g_label.fill_(k)
                    self._g_h.copy_(self._g_h_out)
                    self._g_c.copy_(self._g_c_out)

                symbols_added += 1
                time_idx      += skip
                need_loop      = need_loop and (skip == 0)

            if skip == 0:
                skip = 1
            if symbols_added == 10:
                time_idx += 1

        if return_timestamps:
            return tokens, token_frames, enc_len
        return tokens

    def transcribe(
        self,
        audio: Union[str, 'Path', np.ndarray, torch.Tensor],
        timestamps: bool = False,
    ) -> 'str | TranscriptionResult':
        """Transcribe audio and return text (or a TranscriptionResult with timestamps).

        Args:
            audio:      File path (str/Path), numpy array, or torch Tensor
                        (16 kHz mono float32).
            timestamps: When True, return a TranscriptionResult with char/word/
                        segment-level timestamps instead of a plain string.
        Returns:
            str when timestamps=False (default).
            TranscriptionResult when timestamps=True.
        """
        if self.sp is None:
            raise RuntimeError("No tokenizer loaded. Use from_pretrained() to load the model.")

        if isinstance(audio, (str, Path)):
            from nano_parakeet.audio import convert_to_wav16k, load_audio
            wav_path = convert_to_wav16k(str(audio))
            audio_np = load_audio(wav_path)
            audio_t  = torch.from_numpy(audio_np)
        elif isinstance(audio, np.ndarray):
            audio_t = torch.from_numpy(audio.astype('float32'))
        else:
            audio_t = audio

        if not timestamps:
            token_ids = self.transcribe_audio(audio_t)
            return self.sp.DecodeIds(token_ids).strip()

        result = self.transcribe_audio(audio_t, return_timestamps=True)
        token_ids, token_frames, enc_len = result
        text = self.sp.DecodeIds(token_ids).strip()
        ts   = _frames_to_timestamps(token_ids, token_frames, self.sp, enc_len)
        return TranscriptionResult(text=text, timestamp=ts)
