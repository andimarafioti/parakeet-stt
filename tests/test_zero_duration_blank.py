import types

import torch

from nano_parakeet.model import ParakeetTDT, _frames_to_timestamps, tdt_greedy_decode


class FakeDecoder:
    def __call__(self, last_label, hidden):
        del last_label, hidden
        return torch.zeros(1, 1, 1), None


class FakeJoint:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, enc_out, pred_out):
        del enc_out, pred_out
        return self.logits.view(1, 1, 1, -1)


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


def _blank_logits(blank_id, duration_index=0, duration_count=2):
    token_logits = torch.full((blank_id + 1,), -1000.0)
    token_logits[blank_id] = 1000.0
    dur_logits = torch.full((duration_count,), -1000.0)
    dur_logits[duration_index] = 1000.0
    return torch.cat([token_logits, dur_logits])


def _token_logits(blank_id, token_id, duration_index=1, duration_count=2):
    token_logits = torch.full((blank_id + 1,), -1000.0)
    token_logits[token_id] = 1000.0
    dur_logits = torch.full((duration_count,), -1000.0)
    dur_logits[duration_index] = 1000.0
    return torch.cat([token_logits, dur_logits])


def test_tdt_greedy_decode_advances_on_zero_duration_blank():
    blank_id = 2
    encoder_out = torch.zeros(1, 1)

    result = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=1,
        decoder=FakeDecoder(),
        joint=FakeJoint(_blank_logits(blank_id)),
        blank_id=blank_id,
        durations=[0, 1],
    )

    assert result == []


def test_tdt_decode_graphed_advances_on_zero_duration_blank():
    blank_id = 2
    encoder_out = torch.zeros(1, 1)
    logits = _blank_logits(blank_id)

    model = types.SimpleNamespace(
        DURATIONS=[0, 1],
        BLANK_ID=blank_id,
        _g_h=torch.zeros(1, 1, 1),
        _g_c=torch.zeros(1, 1, 1),
        _g_label=torch.zeros(1, 1, dtype=torch.long),
        _g_f=torch.zeros(1, 1, 1),
        _g_logits=torch.zeros_like(logits),
        _g_h_out=torch.zeros(1, 1, 1),
        _g_c_out=torch.zeros(1, 1, 1),
    )

    class FakeGraph:
        def replay(self):
            model._g_logits.copy_(logits)

    model._decode_graph = FakeGraph()

    result = ParakeetTDT._tdt_decode_graphed(model, encoder_out=encoder_out, enc_len=1)

    assert result == []


def test_tdt_greedy_decode_returns_tokens_and_timestamps():
    blank_id = 2
    encoder_out = torch.zeros(2, 1)

    result = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=2,
        decoder=FakeDecoder(),
        joint=ScriptedJoint(
            [
                _token_logits(blank_id, token_id=1, duration_index=1),
                _blank_logits(blank_id, duration_index=1),
            ]
        ),
        blank_id=blank_id,
        durations=[0, 1],
        return_timestamps=True,
    )

    assert result == ([1], [(0, 1)])


def test_tdt_greedy_decode_handles_zero_duration_label_loop_before_blank():
    blank_id = 2
    encoder_out = torch.zeros(1, 1)

    result = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=1,
        decoder=FakeDecoder(),
        joint=ScriptedJoint(
            [
                _token_logits(blank_id, token_id=1, duration_index=0),
                _blank_logits(blank_id, duration_index=0),
            ]
        ),
        blank_id=blank_id,
        durations=[0, 1],
        return_timestamps=True,
    )

    assert result == ([1], [(0, 0)])


def test_tdt_decode_graphed_matches_scripted_non_graphed_decode():
    blank_id = 2
    encoder_out = torch.zeros(2, 1)
    logits_sequence = [
        _token_logits(blank_id, token_id=1, duration_index=1),
        _blank_logits(blank_id, duration_index=1),
    ]

    non_graphed = tdt_greedy_decode(
        encoder_out=encoder_out,
        enc_len=2,
        decoder=FakeDecoder(),
        joint=ScriptedJoint(logits_sequence),
        blank_id=blank_id,
        durations=[0, 1],
        return_timestamps=True,
    )

    model = types.SimpleNamespace(
        DURATIONS=[0, 1],
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

    graphed = ParakeetTDT._tdt_decode_graphed(
        model,
        encoder_out=encoder_out,
        enc_len=2,
        return_timestamps=True,
    )

    assert graphed == ((*non_graphed, 2))


def test_frames_to_timestamps_extends_zero_duration_tokens():
    sp = FakeSentencePiece({0: "▁hi", 1: "▁there."})

    timestamps = _frames_to_timestamps(
        token_ids=[0, 1],
        token_frames=[(0, 0), (1, 2)],
        sp=sp,
        enc_len=2,
    )

    assert timestamps["word"] == [
        {"word": "hi", "start": 0.0, "end": 0.08},
        {"word": "there.", "start": 0.08, "end": 0.16},
    ]
    assert timestamps["segment"] == [
        {"segment": "hi there.", "start": 0.0, "end": 0.16}
    ]
