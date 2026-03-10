import types

import torch

from nano_parakeet.model import ParakeetTDT, _frames_to_timestamps, tdt_greedy_decode


class FakeDecoder:
    def __call__(self, last_label, hidden):
        del last_label, hidden
        return torch.zeros(1, 1, 1), None


class ScriptedJoint:
    def __init__(self, logits_sequence):
        self.logits_sequence = list(logits_sequence)
        self.index = 0

    def __call__(self, enc_out, pred_out):
        del enc_out, pred_out
        logits = self.logits_sequence[min(self.index, len(self.logits_sequence) - 1)]
        self.index += 1
        return logits.view(1, 1, 1, -1)


class FakeSentencePiece:
    def __init__(self, pieces):
        self.pieces = pieces

    def IdToPiece(self, token_id):
        return self.pieces[token_id]


def _blank_logits(blank_id, duration_index=1, duration_count=3):
    token_logits = torch.full((blank_id + 1,), -1000.0)
    token_logits[blank_id] = 1000.0
    dur_logits = torch.full((duration_count,), -1000.0)
    dur_logits[duration_index] = 1000.0
    return torch.cat([token_logits, dur_logits])


def _token_logits(blank_id, token_id, duration_index=1, duration_count=3):
    token_logits = torch.full((blank_id + 1,), -1000.0)
    token_logits[token_id] = 1000.0
    dur_logits = torch.full((duration_count,), -1000.0)
    dur_logits[duration_index] = 1000.0
    return torch.cat([token_logits, dur_logits])


def _make_graphed_model(blank_id, logits_sequence):
    model = types.SimpleNamespace(
        DURATIONS=[0, 1, 2],
        BLANK_ID=blank_id,
        _g_h=torch.zeros(1, 1, 1),
        _g_c=torch.zeros(1, 1, 1),
        _g_label=torch.zeros(1, 1, dtype=torch.long),
        _g_f=torch.zeros(1, 1, 1),
        _g_logits=torch.zeros_like(logits_sequence[0]),
        _g_h_out=torch.zeros(1, 1, 1),
        _g_c_out=torch.zeros(1, 1, 1),
    )

    class FakeGraph:
        def __init__(self):
            self.index = 0

        def replay(self):
            logits = logits_sequence[min(self.index, len(logits_sequence) - 1)]
            self.index += 1
            model._g_logits.copy_(logits)

    model._decode_graph = FakeGraph()
    return model


def test_tdt_greedy_decode_emits_multiple_tokens_across_frames():
    blank_id = 3
    encoder_out = torch.zeros(4, 1)
    logits_sequence = [
        _token_logits(blank_id, token_id=0, duration_index=1),
        _token_logits(blank_id, token_id=1, duration_index=2),
        _blank_logits(blank_id, duration_index=1),
    ]

    result = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=4,
        decoder=FakeDecoder(),
        joint=ScriptedJoint(logits_sequence),
        blank_id=blank_id,
        durations=[0, 1, 2],
        return_timestamps=True,
    )

    assert result == ([0, 1], [(0, 1), (1, 3)])


def test_tdt_greedy_decode_returns_token_ids_without_timestamps():
    blank_id = 3
    encoder_out = torch.zeros(4, 1)
    logits_sequence = [
        _token_logits(blank_id, token_id=0, duration_index=1),
        _token_logits(blank_id, token_id=1, duration_index=2),
        _blank_logits(blank_id, duration_index=1),
    ]

    result = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=4,
        decoder=FakeDecoder(),
        joint=ScriptedJoint(logits_sequence),
        blank_id=blank_id,
        durations=[0, 1, 2],
    )

    assert result == [0, 1]


def test_tdt_decode_graphed_matches_non_graphed_on_standard_sequence():
    blank_id = 3
    encoder_out = torch.zeros(4, 1)
    logits_sequence = [
        _token_logits(blank_id, token_id=0, duration_index=1),
        _token_logits(blank_id, token_id=1, duration_index=2),
        _blank_logits(blank_id, duration_index=1),
    ]

    non_graphed = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=4,
        decoder=FakeDecoder(),
        joint=ScriptedJoint(logits_sequence),
        blank_id=blank_id,
        durations=[0, 1, 2],
        return_timestamps=True,
    )

    model = _make_graphed_model(blank_id, logits_sequence)

    graphed = ParakeetTDT._tdt_decode_graphed(
        model,
        encoder_out=encoder_out,
        enc_len=4,
        return_timestamps=True,
    )

    assert graphed == ((*non_graphed, 4))


def test_frames_to_timestamps_builds_words_segments_and_chars():
    timestamps = _frames_to_timestamps(
        token_ids=[0, 1, 2],
        token_frames=[(0, 1), (1, 2), (2, 4)],
        sp=FakeSentencePiece({0: "▁Hel", 1: "lo", 2: "▁world!"}),
        enc_len=4,
    )

    assert "".join(entry["char"] for entry in timestamps["char"]) == "Hello world!"
    assert timestamps["word"] == [
        {"word": "Hello", "start": 0.0, "end": 0.16},
        {"word": "world!", "start": 0.16, "end": 0.32},
    ]
    assert timestamps["segment"] == [
        {"segment": "Hello world!", "start": 0.0, "end": 0.32}
    ]
