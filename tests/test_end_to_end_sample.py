from pathlib import Path

from nano_parakeet import from_pretrained


EXPECTED_SAMPLE_TRANSCRIPT = (
    "I'm confused why some people have super short timelines, yet at the same "
    "time are bullish on scaling up reinforcement learning atop LLMs. If we're "
    "actually close to a human-like learner, then this whole approach of "
    "training on verifiable outcomes."
)


def test_sample_wav_full_transcription_matches_expected_output():
    sample_path = Path(__file__).with_name("sample.wav")

    model = from_pretrained(device="cpu")
    transcript = model.transcribe(str(sample_path))

    assert transcript == EXPECTED_SAMPLE_TRANSCRIPT
