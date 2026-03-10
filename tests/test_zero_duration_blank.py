import types

import torch

from nano_parakeet.model import ParakeetTDT, tdt_greedy_decode


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


def _logits(blank_id, duration_index=0, duration_count=2):
    token_logits = torch.full((blank_id + 1,), -1000.0)
    token_logits[blank_id] = 1000.0
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
        joint=FakeJoint(_logits(blank_id)),
        blank_id=blank_id,
        durations=[0, 1],
    )

    assert result == []


def test_tdt_decode_graphed_advances_on_zero_duration_blank():
    blank_id = 2
    encoder_out = torch.zeros(1, 1)
    logits = _logits(blank_id)

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
