# STTTS-Multi + DiCoW pipeline

Alternative to [STTTS-Multi](../sttts_multi/readme_sttts_multi_pipeline.md) for **multi-speaker English dialogues**.
Speaker `0` is GAN-anonymized (exclusive target and overlap). Exclusive non-target speech and silence stay original. Overlaps are mixed from anonymized target TTS and non-target TTS (original embeddings, no GAN) at the original overlap energy.

DiCoW is a diarization-conditioned Whisper ([BUT-FIT/DiCoW_v3_2](https://huggingface.co/BUT-FIT/DiCoW_v3_2)).
It does not replace an external diarizer: it consumes **lhotse supervisions** as speaker activity and turns them into STNO masks (silence / target / non-target / overlap).

Weights are loaded from the hub; the Python classes are the transformers-4.55 port under `anonymization/modules/sttts_multi/text/recognition/dicow_model/` (HF `trust_remote_code` for this checkpoint is 4.42-only). Extra requirements pin `transformers==4.55.0` in `venv_sttts_multi`.

## What is different from STTTS-Multi (BM2/BM3)

| Step | STTTS-Multi | This pipeline |
| --- | --- | --- |
| Input | Kaldi dir (`wav.scp`, `utt2spk`) of single-speaker utterances | Lhotse cutset of **mixtures** + diarization supervisions |
| ASR | `openai/whisper-large-v3` on each file independently | `BUT-FIT/DiCoW_v3_2` on the mix, once per speaker (features reused) |
| Who is transcribed | Every utterance | Every speaker on cuts that contain `target_speaker_id` (default `"0"`) |
| Speaker embedding | Extracted from the full utterance wav | One ECAPA on concatenated exclusive speech per (mix, speaker); fallback: all that speaker's turns if exclusive < 2s |
| Who is anonymized | Every speaker | Only speaker `"0"` (GAN). Others keep original ECAPA |
| TTS output | One anonymized wav per utterance | One TTS wav per speaker turn |
| Final audio | TTS wavs | Original mix: exclusive target → TTS; exclusive non-target + silence kept; overlaps mixed at original energy |

Speaker embedding, GAN anonymization, and IMS Toucan TTS are the same modules as in STTTS-Multi.
Prosody is on (BM2-style): the IMS aligner estimates per-phone duration/pitch/energy on a mix crop of each turn; Toucan synthesizes with those durations. Remix only pads or crops leftover samples (no librosa stretch).

Cuts that do not contain the target speaker are copied unchanged.

## Remix

* exclusive **target** → anonymized TTS (scaled to original exclusive RMS)
* exclusive **non-target** → original mix
* **silence** → original mix
* **overlap** → sum of target TTS + non-target TTS, then short-time RMS matched to the original overlap

Non-target TTS is synthesized for the full turn (so the overlap slice is time-aligned), but only the overlap frames are written back.

## Usage

```bash
python run_anonymization.py \
  --config configs/track2/anon_sttts_multi_dicow.yaml \
  --gpu_ids 0
```

Point `datasets[].lhotse_cutset` at your saved lhotse manifest (`.jsonl.gz` cutset with `speaker` on each supervision).
If you change `datasets[].data`, also add that name to `modules.speaker_embeddings.anon_level_spk` (or `anon_level_utt`).

Outputs:

* original mix kaldi dir: `data/<dataset>/`
* remixed anonymized mixes: `data/<dataset>_sttts_dicow/`
* DiCoW transcripts: `exp/anon_pipeline_sttts_multi_dicow/transcription/dicow/<dataset>/`
* per-speaker crops used for embeddings: `exp/anon_pipeline_sttts_multi_dicow/all_speaker_kaldi/<dataset>/`
* per-turn crops for the aligner: `exp/anon_pipeline_sttts_multi_dicow/turn_kaldi/<dataset>/`

Pitch is not randomly offset (unlike BM2 `anonymizer_type: ims`). Set `modules.tts.force_compute_synthesis` to `false` after the first run with gold durations.
