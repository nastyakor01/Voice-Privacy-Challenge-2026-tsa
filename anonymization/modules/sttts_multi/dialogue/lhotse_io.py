"""Helpers for DiCoW-conditioned STTTS on lhotse dialogue cutsets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile
from lhotse import CutSet, MonoCut, Recording, load_manifest
from lhotse.utils import fastcopy

from utils import read_kaldi_format, save_kaldi_format, setup_logger

logger = setup_logger(__name__)

# SpeechBrain ECAPA-TDNN needs enough fbank frames (20 ms crops crash in conv padding).
MIN_ECAPA_EMBED_SEC = 0.5
# IMS aligner / AudioPreprocessor fail on near-empty turn crops.
MIN_TURN_CROP_SEC = 0.1

TIMESTAMP_PAIR_RE = re.compile(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>", re.DOTALL) # looks for timestamp pairs and the text between them. Example: <|1.234|>Hello<|5.678|>
TIMESTAMP_TOKEN_RE = re.compile(r"<\|[\d.]+\|>") # looks for timestamp tokens. Example: <|1.234|>


def load_cutset(path) -> CutSet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lhotse cutset not found: {path}")
    manifest = load_manifest(path)
    if hasattr(manifest, "to_eager"):
        return manifest.to_eager()
    return manifest


def cut_speakers(cut) -> list[str]:
    return sorted({str(sup.speaker) for sup in cut.supervisions if sup.speaker is not None})


def cut_has_speaker(cut, speaker_id: str) -> bool:
    return str(speaker_id) in cut_speakers(cut)


def cut_anonymize_speaker(cut, default: str | None = "0") -> str | None:
    """Speaker id to GAN-anonymize, from ``cut.custom['anonymize']``.

    The value is the mix-local speaker label (e.g. ``"0"``, ``"1"``). Falls back to ``default`` if custom is
    missing. Returns ``None`` if that speaker is not on the cut.
    """
    custom = getattr(cut, "custom", None) or {}
    val = custom.get("anonymize")
    if val is None or val == "":
        speaker_id = None if default is None else str(default)
    else:
        speaker_id = str(val)
    if speaker_id is None:
        return None
    if not cut_has_speaker(cut, speaker_id):
        return None
    return speaker_id


def load_cut_audio(cut) -> tuple[np.ndarray, int]:
    audio = cut.load_audio()
    sr = cut.sampling_rate
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio[0] if audio.shape[0] <= audio.shape[1] else audio[:, 0]
    return np.ascontiguousarray(audio.squeeze()), int(sr)


def speakers_sample_mask(cut, speaker_ids: list[str] | None = None) -> np.ndarray:
    """Binary mask of shape (n_speakers, n_samples) from lhotse supervisions."""
    audio, sr = load_cut_audio(cut)
    n_samples = audio.shape[-1]
    speaker_ids = speaker_ids or cut_speakers(cut)
    spk2idx = {spk: i for i, spk in enumerate(speaker_ids)}
    mask = np.zeros((len(speaker_ids), n_samples), dtype=np.float32)
    cut_start = float(cut.start)
    for sup in cut.supervisions:
        spk = str(sup.speaker)
        if spk not in spk2idx:
            continue
        start = int(round((float(sup.start) - cut_start) * sr))
        end = int(round((float(sup.start) + float(sup.duration) - cut_start) * sr))
        start = max(0, min(n_samples, start))
        end = max(start, min(n_samples, end))
        mask[spk2idx[spk], start:end] = 1.0
    return mask


def _create_stno_masks(spk_mask: np.ndarray, s_index: int) -> np.ndarray:
    non_target_mask = np.ones(spk_mask.shape[0], dtype=bool)
    non_target_mask[s_index] = False
    sil_frames = (1 - spk_mask).prod(axis=0)
    if spk_mask.shape[0] == 1:
        anyone_else = np.ones(spk_mask.shape[1], dtype=np.float32) # no one else speaks except the target speaker
    else:
        anyone_else = (1 - spk_mask[non_target_mask]).prod(axis=0) # no one else speaks except the target speaker
    target_spk = spk_mask[s_index] * anyone_else
    non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
    overlapping_speech = spk_mask[s_index] - target_spk
    return np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)


def build_stno_mask(cut, speaker_id: str, n_enc_frames: int, n_samples_padded: int) -> np.ndarray:
    """STNO mask (4, n_enc_frames) at Whisper encoder rate (~50 Hz).

    Channels: silence, target, non-target, overlap. Matches DiCoW / TS-ASR.
    """
    speaker_ids = cut_speakers(cut)
    speaker_id = str(speaker_id)
    if speaker_id not in speaker_ids:
        raise KeyError(f"Speaker {speaker_id} not in cut {cut.id}: {speaker_ids}")

    spk_mask = speakers_sample_mask(cut, speaker_ids)
    pad_len = max(0, n_samples_padded - spk_mask.shape[-1])
    if pad_len:
        spk_mask = np.pad(spk_mask, ((0, 0), (0, pad_len)), mode="constant")
    elif spk_mask.shape[-1] > n_samples_padded:
        spk_mask = spk_mask[:, :n_samples_padded]

    frames_per_enc = max(1, n_samples_padded // n_enc_frames)
    usable = (spk_mask.shape[-1] // frames_per_enc) * frames_per_enc
    spk_mask = spk_mask[:, :usable]
    spk_mask = spk_mask.reshape(spk_mask.shape[0], -1, frames_per_enc).mean(axis=-1)

    stno = _create_stno_masks(spk_mask, speaker_ids.index(speaker_id)).astype(np.float32)
    if stno.shape[-1] < n_enc_frames:
        pad = np.zeros((4, n_enc_frames - stno.shape[-1]), dtype=np.float32)
        pad[0] = 1.0
        stno = np.concatenate([stno, pad], axis=-1)
    elif stno.shape[-1] > n_enc_frames:
        stno = stno[:, :n_enc_frames]
    return stno


def build_all_target_stno_mask(n_enc_frames: int) -> np.ndarray:
    """STNO with the whole clip as target; no supervision / diarization clock."""
    n = max(1, int(n_enc_frames))
    stno = np.zeros((4, n), dtype=np.float32)
    stno[1] = 1.0
    return stno


def target_supervision_windows(cut, speaker_id: str) -> list[tuple[float, float]]:
    """Start/end of each speaker turn from lhotse supervisions, sorted."""
    windows = []
    for sup in cut.supervisions:
        if str(sup.speaker) != str(speaker_id):
            continue
        start = float(sup.start)
        end = start + float(sup.duration)
        if end > start:
            windows.append((start, end))
    windows.sort(key=lambda w: w[0])
    return windows


def cut_speaker_windows(cut) -> dict[str, list[list[float]]]:
    """Lhotse supervision windows per speaker, used as remix activity."""
    windows: dict[str, list[list[float]]] = {}
    for spk in cut_speakers(cut):
        windows[spk] = [[w0, w1] for w0, w1 in target_supervision_windows(cut, spk)]
    return windows


def cut_audio_window(cut) -> tuple[float, float] | None:
    """Full cut audio on the same timeline as supervisions: [cut.start, cut.start + duration]."""
    start = float(getattr(cut, "start", 0.0) or 0.0)
    end = start + float(getattr(cut, "duration", 0.0) or 0.0)
    if end <= start:
        return None
    return start, end


def _merge_intervals(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse overlapping / touching intervals so subtraction sees each time once.

    Other speakers can have adjacent or overlapping turns; without a merge,
    ``_subtract_intervals`` would punch the same region twice and miss gaps.
    """
    items = sorted((float(a), float(b)) for a, b in windows if b > a)
    if not items:
        return []
    merged = [items[0]]
    for start, end in items[1:]:
        last_s, last_e = merged[-1]
        if start <= last_e:
            merged[-1] = (last_s, max(last_e, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    bases: list[tuple[float, float]], others: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Keep only the parts of ``bases`` that do not overlap ``others``.

    Used to cut other speakers' activity out of this speaker's turns, leaving
    exclusive speech for the ECAPA crop.
    """
    others = _merge_intervals(others)
    out = []
    for start, end in bases:
        pieces = [(start, end)]
        for o0, o1 in others:
            next_pieces = []
            for p0, p1 in pieces:
                if o1 <= p0 or o0 >= p1:
                    next_pieces.append((p0, p1))
                    continue
                if p0 < o0:
                    next_pieces.append((p0, o0))
                if o1 < p1:
                    next_pieces.append((o1, p1))
            pieces = next_pieces
        out.extend((a, b) for a, b in pieces if b > a)
    return out


def exclusive_supervision_windows(cut, speaker_id: str) -> list[tuple[float, float]]:
    """Lhotse turns of ``speaker_id`` minus any overlap with other speakers.

    ECAPA should see this speaker alone when possible: overlap is a mixture and
    would pull the embedding toward the other voice.
    """
    own = target_supervision_windows(cut, speaker_id)
    others = []
    for spk in cut_speakers(cut):
        if str(spk) != str(speaker_id):
            others.extend(target_supervision_windows(cut, spk))
    return _subtract_intervals(own, others)


def mix_speaker_embedding_utt(cut_id: str, speaker_id: str) -> str:
    """Stable Kaldi / embedding id for the one concat crop of this speaker in the mix.

    ``cut_id`` is ``cut.id``, e.g. SparseLibriMix ``mix_0000001``. Distinct from
    ASR/TTS ids (``mix_0000001-0-0001``): this utt is only for ECAPA, then the
    vector is copied onto every TTS turn of that speaker.
    Example: ``mix_0000001__spk0_excl``.
    """
    return f"{cut_id}__spk{speaker_id}_excl"


def speaker_mix_embedding_windows(
    cut, speaker_id: str, min_exclusive_sec: float = 1
) -> list[list[float]]:
    """Time ranges to concatenate into the single ECAPA crop for this speaker.

    Prefer all exclusive speech in the mix if it totals at least
    ``min_exclusive_sec``. If not, exclusive audio is too short for a stable
    embedding, so fall back to concatenating every turn (including overlap).
    """
    exclusive = exclusive_supervision_windows(cut, speaker_id)
    if sum(end - start for start, end in exclusive) >= min_exclusive_sec:
        return [[start, end] for start, end in exclusive]
    return [[start, end] for start, end in target_supervision_windows(cut, speaker_id)]


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def hyp_to_target_turn_segments(hyp_text: str, windows: list[tuple[float, float]]) -> list[dict]:
    """Map a DiCoW hypothesis onto lhotse target-speaker turns.

    ``windows`` are this speaker's supervision turns. Placement is always those
    turns. Timestamps, if present, only decide which turn gets which words —
    and only when every window receives some text. If timestamps leave any
    turn empty (typical: one long ``<|t|>…<|t|>`` blob vs two or more GT turns), do
    not drop the empty turns: one TTS utterance is split across all windows
    in remix, same as the no-timestamp path.
    """
    if not windows:
        return []
    phrases = []
    for start, body, end in TIMESTAMP_PAIR_RE.findall(hyp_text or ""): # find all timestamp pairs and the text between them. Example: <|1.234|>Hello<|5.678|> -> ("1.234", "Hello", "5.678")
        body = body.strip()
        if body:
            phrases.append((float(start), float(end), body))
    full_text = TIMESTAMP_TOKEN_RE.sub("", hyp_text or "").strip() # remove timestamps from the phrase: "Hello <|1.234|> World <|5.678|>" -> "Hello  World "

    if phrases:
        assigned = [[] for _ in windows]
        for t0, t1, body in phrases:
            overlaps = [_interval_overlap(t0, t1, w0, w1) for w0, w1 in windows]
            if max(overlaps) > 0:
                j = int(max(range(len(windows)), key=lambda i: overlaps[i]))
            else:
                mid = 0.5 * (t0 + t1)
                j = min(range(len(windows)),
                        key=lambda i: abs(0.5 * (windows[i][0] + windows[i][1]) - mid))
            assigned[j].append(body)
        if all(" ".join(bits).strip() for bits in assigned): # if no windows are empty
            return [{
                "start": w0,
                "end": w1,
                "text": " ".join(bits).strip(),
                "placement_windows": [[w0, w1]],
            } for (w0, w1), bits in zip(windows, assigned)]
        # Incomplete timestamp coverage: fall through to one TTS on all windows.

    return hyp_to_concat_segment(hyp_text, windows)


def hyp_to_concat_segment(hyp_text: str, windows: list[tuple[float, float]]) -> list[dict]:
    """One TTS; aligner and remix concatenate every supervision turn. Ignore timestamps."""
    if not windows:
        return []
    full_text = TIMESTAMP_TOKEN_RE.sub("", hyp_text or "").strip() # gremove timestamps from dicow output:  <|0.00|>hello world<|2.40|> becomes hello world
    if not full_text:
        return []
    placed = [[float(w0), float(w1)] for w0, w1 in windows if float(w1) > float(w0)]
    if not placed:
        return []
    return [{
        "start": placed[0][0],
        "end": placed[-1][1],
        "text": full_text,
        "placement_windows": placed,
    }]


def hyp_to_span_segment(hyp_text: str, windows: list[tuple[float, float]]) -> list[dict]:
    """One TTS over [min start, max end] of ``windows``; ignore Whisper timestamps."""
    if not windows:
        return []
    full_text = TIMESTAMP_TOKEN_RE.sub("", hyp_text or "").strip()
    if not full_text:
        return []
    w0 = min(float(w[0]) for w in windows)
    w1 = max(float(w[1]) for w in windows)
    if w1 <= w0:
        return []
    return [{
        "start": w0,
        "end": w1,
        "text": full_text,
        "placement_windows": [[w0, w1]],
    }]


def _crop_windows(audio: np.ndarray, sr: int, cut_start: float,
                  windows: list) -> np.ndarray:
    n = audio.shape[-1]
    crops = []
    for window in windows:
        w0, w1 = float(window[0]), float(window[1])
        start = max(0.0, w0 - cut_start)
        end = max(start, w1 - cut_start)
        start_i = min(max(0, int(round(start * sr))), n)
        end_i = min(max(start_i, int(round(end * sr))), n)
        if end_i > start_i:
            crops.append(audio[start_i:end_i])
    if not crops:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(crops)


def audio_duration_sec(audio: np.ndarray, sr: int) -> float:
    if audio is None or getattr(audio, "size", 0) == 0 or sr <= 0:
        return 0.0
    return float(audio.size) / float(sr)


def _skip_short_ecapa_crop(
    audio: np.ndarray,
    sr: int,
    min_embed_sec: float,
    mix_utt: str,
    source: str,
) -> bool:
    """True if this crop is too short for a stable ECAPA call."""
    dur = audio_duration_sec(audio, sr)
    if min_embed_sec <= 0 or dur >= min_embed_sec:
        return False
    logger.warning(
        f"Skip ECAPA {mix_utt}: {dur:.3f}s < {min_embed_sec}s ({source}); "
        "no TTS for this speaker, remix keeps original"
    )
    return True


def write_mix_kaldi_dir(cutset: CutSet, out_dir: Path, default_gender: str = "m") -> Path:
    """One kaldi utterance per mix/cut (original mixture wavs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_scp, utt2spk, spk2gender = {}, {}, {}
    spk2utts = {}
    skipped = 0
    for cut in cutset:
        custom = getattr(cut, "custom", None) or {}
        speaker = custom.get("libri_spk_id")
        if speaker is None or speaker == "":
            skipped += 1
            continue
        speaker = str(speaker)
        wav_path = Path(cut.recording.sources[0].source).resolve()
        utt = str(cut.id)
        wav_scp[utt] = str(wav_path)
        utt2spk[utt] = speaker
        spk2utts.setdefault(speaker, []).append(utt)
        spk2gender[speaker] = default_gender
    if skipped:
        logger.warning(f"write_mix_kaldi_dir: skipped {skipped} cuts without custom['anonymize']")
    save_kaldi_format(wav_scp, out_dir / "wav.scp")
    save_kaldi_format(utt2spk, out_dir / "utt2spk")
    save_kaldi_format(spk2gender, out_dir / "spk2gender")
    save_kaldi_format({spk: " ".join(sorted(utts)) for spk, utts in spk2utts.items()}, out_dir / "spk2utt")
    return out_dir


def write_cutset_rttm(cutset: CutSet, out_path: Path) -> Path:
    """Write input supervisions as RTTM (same clock used for concat TTS / remix)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cut in cutset:
        reco = str(cut.id)
        for sup in cut.supervisions:
            if sup.speaker is None:
                continue
            start = float(sup.start)
            dur = float(sup.duration)
            if dur <= 0:
                continue
            lines.append(
                f"SPEAKER {reco} 1 {start:.3f} {dur:.3f} <NA> <NA> {sup.speaker} <NA> <NA>"
            )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def write_speaker_segment_kaldi_dir(
    cutset: CutSet,
    segments: list[dict],
    out_dir: Path,
    wav_dir: Path,
    default_gender: str = "m",
    min_exclusive_sec: float = 2.0,
    full_cut: bool = False,
    min_embed_sec: float = MIN_ECAPA_EMBED_SEC,
) -> Path:
    """Write one concatenated crop per (mix, speaker) for a single ECAPA call.

    If ``full_cut``, the crop is the entire cut audio (TSE s1 after attach).
    Else exclusive speech if >= ``min_exclusive_sec``, otherwise all turns.
    Crops shorter than ``min_embed_sec`` are skipped (no ECAPA/TTS; remix original).
    """
    out_dir = Path(out_dir)
    wav_dir = Path(wav_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    cuts = {str(cut.id): cut for cut in cutset}
    wav_scp, utt2spk, utt2lang = {}, {}, {}
    skipped = []
    mix_pairs = {
        (str(seg["cut_id"]), str(seg.get("speaker")))
        for seg in segments
        if str(seg["cut_id"]) in cuts and seg.get("speaker") is not None
    }
    # cut_id == cut.id, e.g. "mix_0000001" (one dialogue mixture).
    for cut_id, speaker_id in sorted(mix_pairs):
        cut = cuts[cut_id]
        audio, sr = load_cut_audio(cut)
        if full_cut:
            window = cut_audio_window(cut)
            windows = [window] if window else []
            source = "full_cut"
        else:
            windows = speaker_mix_embedding_windows(cut, speaker_id, min_exclusive_sec)
            source = "exclusive_or_turns"
        crop = _crop_windows(audio, sr, float(cut.start), windows)
        mix_utt = mix_speaker_embedding_utt(cut_id, speaker_id)
        if crop.size == 0 or _skip_short_ecapa_crop(crop, sr, min_embed_sec, mix_utt, source):
            skipped.append({
                "cut_id": cut_id,
                "speaker": speaker_id,
                "mix_utt": mix_utt,
                "source": source,
                "duration_sec": audio_duration_sec(crop, sr),
            })
            continue
        wav_path = wav_dir / f"{mix_utt}.wav"
        soundfile.write(str(wav_path), crop, sr)
        wav_scp[mix_utt] = str(wav_path.resolve())
        utt2spk[mix_utt] = speaker_id
        utt2lang[mix_utt] = "en"

    if not wav_scp:
        raise RuntimeError("No speaker segments could be written (empty DiCoW output?)")

    _dump_speaker_kaldi_tables(out_dir, wav_scp, utt2spk, utt2lang, segments, default_gender)
    _dump_skipped_short_ecapa(out_dir, skipped)
    return out_dir


def load_tse_enrolls_index(tse_support_enrolls_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """TSE support-enroll wav.scp / utt2spk: LibriSpeech utt → path, speaker → enroll utt."""
    tse_support_enrolls_dir = Path(tse_support_enrolls_dir)
    wav_scp_path = tse_support_enrolls_dir / "wav.scp"
    utt2spk_path = tse_support_enrolls_dir / "utt2spk"
    wav_by_utt: dict[str, Path] = {}
    if wav_scp_path.exists():
        for utt, path in read_kaldi_format(wav_scp_path).items():
            wav_by_utt[str(utt)] = Path(path)
    spk_to_utt: dict[str, str] = {}
    if utt2spk_path.exists():
        for utt, spk in read_kaldi_format(utt2spk_path).items():
            spk_to_utt.setdefault(str(spk), str(utt))
    else:
        for utt in wav_by_utt:
            spk_to_utt.setdefault(str(utt).split("-")[0], str(utt))
    return wav_by_utt, spk_to_utt


def load_mix_enroll_scp(source) -> dict[str, str]:
    """TSE ``s1/enroll.scp``: mix id → LibriSpeech enroll utterance."""
    path = Path(source) / "enroll.scp" if Path(source).is_dir() else Path(source)
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in read_kaldi_format(path).items()}


def resolve_tse_enroll_utt(
    cut,
    wav_by_utt: dict[str, Path],
    mix_to_enroll: dict[str, str] | None = None,
    speaker_id: str | None = None,
    target_speaker: str | None = None,
) -> tuple[str | None, str]:
    """TSE support enroll for the anonymize speaker only; others use exclusive/turns."""
    if speaker_id is None or target_speaker is None or str(speaker_id) != str(target_speaker):
        return None, "exclusive_or_turns"
    cut_id = str(cut.id)
    mix_to_enroll = mix_to_enroll or {}
    if cut_id not in mix_to_enroll:
        return None, "exclusive_or_turns"
    utt = str(mix_to_enroll[cut_id])
    enroll_path = wav_by_utt.get(utt)
    if enroll_path is None or not Path(enroll_path).exists():
        return None, "exclusive_or_turns"
    return utt, "enroll.scp"


def write_tse_enroll_embedding_kaldi_dir(
    cutset: CutSet,
    segments: list[dict],
    out_dir: Path,
    wav_dir: Path,
    tse_support_enrolls_dir: Path,
    mix_to_enroll: dict[str, str] | None = None,
    default_gender: str = "m",
    min_exclusive_sec: float = 2.0,
    min_embed_sec: float = MIN_ECAPA_EMBED_SEC,
) -> Path:
    """Write ECAPA crops from TSE enroll wavs, else exclusive/turns of ``cutset``.

    Pass the original mix cutset (before TSE s1 attach). Ids stay
    ``mix_speaker_embedding_utt(cut_id, speaker)`` so GAN lookup is unchanged.
    Crops shorter than ``min_embed_sec`` are skipped (no ECAPA/TTS; remix original).
    """
    out_dir = Path(out_dir)
    wav_dir = Path(wav_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    wav_by_utt, _spk_to_utt = load_tse_enrolls_index(tse_support_enrolls_dir)
    mix_to_enroll = mix_to_enroll or {}

    cuts = {str(cut.id): cut for cut in cutset}
    wav_scp, utt2spk, utt2lang = {}, {}, {}
    mix_pairs = {
        (str(seg["cut_id"]), str(seg.get("speaker")))
        for seg in segments
        if str(seg["cut_id"]) in cuts and seg.get("speaker") is not None
    }
    map_records = []
    skipped = []
    n_enroll = 0
    n_excl = 0
    n_target_excl = 0
    n_skip_short = 0
    for cut_id, speaker_id in sorted(mix_pairs):
        cut = cuts[cut_id]
        target = cut_anonymize_speaker(cut, default=None)
        enroll_utt, source = resolve_tse_enroll_utt(
            cut, wav_by_utt, mix_to_enroll, speaker_id=speaker_id, target_speaker=target
        ) # find tse enroll utt for the target speaker. for the non-target speakers, use exclusive/turns.
        enroll_path = wav_by_utt.get(enroll_utt) if enroll_utt else None
        if source == "enroll.scp" and enroll_path is not None:
            audio, sr = _load_wav_file(Path(enroll_path))
        else:
            audio, sr = load_cut_audio(cut)
            windows = speaker_mix_embedding_windows(cut, speaker_id, min_exclusive_sec)
            audio = _crop_windows(audio, sr, float(cut.start), windows)
            source = "exclusive_or_turns"
        mix_utt = mix_speaker_embedding_utt(cut_id, speaker_id)
        if audio.size == 0 or _skip_short_ecapa_crop(audio, sr, min_embed_sec, mix_utt, source):
            n_skip_short += 1
            skipped.append({
                "cut_id": cut_id,
                "speaker": speaker_id,
                "mix_utt": mix_utt,
                "enroll_utt": enroll_utt,
                "enroll_wav": str(enroll_path) if enroll_path is not None else None,
                "source": source,
                "duration_sec": audio_duration_sec(audio, sr),
            })
            map_records.append({
                "cut_id": cut_id,
                "speaker": speaker_id,
                "mix_utt": mix_utt,
                "enroll_utt": enroll_utt,
                "enroll_wav": str(enroll_path) if enroll_path is not None else None,
                "source": f"{source}_skipped_short",
                "duration_sec": audio_duration_sec(audio, sr),
            })
            continue
        if source == "enroll.scp":
            n_enroll += 1
        else:
            n_excl += 1
            if target is not None and str(speaker_id) == str(target):
                n_target_excl += 1
        wav_path = wav_dir / f"{mix_utt}.wav"
        soundfile.write(str(wav_path), audio, sr)
        wav_scp[mix_utt] = str(wav_path.resolve())
        utt2spk[mix_utt] = speaker_id
        utt2lang[mix_utt] = "en"
        map_records.append({
            "cut_id": cut_id,
            "speaker": speaker_id,
            "mix_utt": mix_utt,
            "enroll_utt": enroll_utt,
            "enroll_wav": str(enroll_path) if enroll_path is not None else None,
            "source": source,
            "duration_sec": audio_duration_sec(audio, sr),
        })

    if not wav_scp:
        raise RuntimeError("No TSE-enroll speaker embeddings could be written")
    _dump_speaker_kaldi_tables(out_dir, wav_scp, utt2spk, utt2lang, segments, default_gender)
    _dump_skipped_short_ecapa(out_dir, skipped)
    with (out_dir / "tse_enroll_map.jsonl").open("w", encoding="utf-8") as f:
        for row in map_records:
            f.write(json.dumps(row) + "\n")
    if n_target_excl:
        logger.warning(
            f"write_tse_enroll_embedding_kaldi_dir: {n_target_excl} target speakers had no TSE enroll, "
            "used exclusive/turns"
        )
    logger.info(
        f"ECAPA crops from tse support enrolls ({tse_support_enrolls_dir}): {len(wav_scp)} utts "
        f"(enroll.scp={n_enroll}, exclusive/turns={n_excl}, skipped_short={n_skip_short}), "
        f"map at {out_dir / 'tse_enroll_map.jsonl'}"
    )
    return out_dir


def _dump_speaker_kaldi_tables(
    out_dir: Path,
    wav_scp: dict,
    utt2spk: dict,
    utt2lang: dict,
    segments: list[dict],
    default_gender: str,
) -> None:
    spk2utts = {}
    for utt, spk in utt2spk.items():
        spk2utts.setdefault(spk, []).append(utt)
    save_kaldi_format(wav_scp, out_dir / "wav.scp")
    save_kaldi_format(utt2spk, out_dir / "utt2spk")
    save_kaldi_format({spk: default_gender for spk in spk2utts}, out_dir / "spk2gender")
    save_kaldi_format({spk: " ".join(sorted(utts)) for spk, utts in spk2utts.items()}, out_dir / "spk2utt")
    save_kaldi_format(utt2lang, out_dir / "utt2lang")
    with (out_dir / "segments.jsonl").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg) + "\n")


def _dump_skipped_short_ecapa(out_dir: Path, skipped: list[dict]) -> None:
    if not skipped:
        return
    path = Path(out_dir) / "skipped_short_ecapa.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in skipped:
            f.write(json.dumps(row) + "\n")
    logger.warning(
        f"Skipped {len(skipped)} short ECAPA crops (< min_embed_sec); "
        f"TTS omitted, remix keeps original. List: {path}"
    )


def write_turn_kaldi_dir(
    cutset: CutSet,
    segments: list[dict],
    out_dir: Path,
    wav_dir: Path,
    default_gender: str = "m",
    load_audio=None,
) -> Path:
    """One crop per ASR/TTS turn, ids matching ``texts`` / DiCoW ``utt``.

    Used by the IMS aligner to get per-phone durations for Toucan (BM2-style).
    By default crops the original mix. Pass ``load_audio(cut, speaker_id)`` to
    crop a different full-length recording (e.g. TSE s1 / residual) at the same
    ``placement_windows``.
    """
    out_dir = Path(out_dir)
    wav_dir = Path(wav_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    cuts = {str(cut.id): cut for cut in cutset}
    wav_scp, utt2spk, utt2lang = {}, {}, {}
    audio_cache = {}
    for seg in segments:
        utt = seg.get("utt")
        cut_id = str(seg.get("cut_id", ""))
        speaker_id = str(seg.get("speaker", ""))
        if not utt or cut_id not in cuts:
            continue
        cut = cuts[cut_id]
        cache_key = (cut_id, speaker_id) if load_audio is not None else cut_id
        if cache_key not in audio_cache:
            if load_audio is None:
                audio_cache[cache_key] = load_cut_audio(cut)
            else:
                audio_cache[cache_key] = load_audio(cut, speaker_id)
        audio, sr = audio_cache[cache_key]
        windows = seg.get("placement_windows") or [[seg["start"], seg["end"]]]
        crop = _crop_windows(audio, sr, float(cut.start), windows)
        if crop.size == 0 or audio_duration_sec(crop, sr) < MIN_TURN_CROP_SEC:
            if crop.size:
                logger.warning(
                    f"Skip turn crop {utt}: {audio_duration_sec(crop, sr):.3f}s "
                    f"< {MIN_TURN_CROP_SEC}s (aligner); remix keeps original if no TTS"
                )
            continue
        wav_path = wav_dir / f"{utt}.wav"
        soundfile.write(str(wav_path), crop, sr)
        wav_scp[utt] = str(wav_path.resolve())
        utt2spk[utt] = speaker_id
        utt2lang[utt] = str(seg.get("language") or "en")

    if not wav_scp:
        raise RuntimeError("No per-turn crops could be written (empty DiCoW output?)")

    spk2utts = {}
    for utt, spk in utt2spk.items():
        spk2utts.setdefault(spk, []).append(utt)
    save_kaldi_format(wav_scp, out_dir / "wav.scp")
    save_kaldi_format(utt2spk, out_dir / "utt2spk")
    save_kaldi_format({spk: default_gender for spk in spk2utts}, out_dir / "spk2gender")
    save_kaldi_format({spk: " ".join(sorted(utts)) for spk, utts in spk2utts.items()}, out_dir / "spk2utt")
    save_kaldi_format(utt2lang, out_dir / "utt2lang")
    return out_dir


_TSE_S1_ALIASES = ("s1", "target")
_TSE_RESIDUAL_ALIASES = ("s1_project", "s1_mask", "s1_residual", "s1_resdial", "residual")


def resolve_tse_dataset_dir(tse_root: Path, dataset_name: str) -> Path | None:
    """Locate ``{tse_root}/{dataset}``; strip ``_mono`` / ``_gt`` if needed."""
    tse_root = Path(tse_root)
    names = [dataset_name]
    for suffix in ("_mono", "_gt"):
        if dataset_name.endswith(suffix):
            names.append(dataset_name[: -len(suffix)])
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        path = tse_root / name
        if not path.is_dir():
            continue
        if any((path / alias).is_dir() for alias in _TSE_S1_ALIASES + _TSE_RESIDUAL_ALIASES):
            return path
    return None


def _resolve_tse_subdir(tse_dataset_dir: Path, preferred: str, aliases: tuple[str, ...]) -> Path | None:
    for name in (preferred,) + aliases:
        if not name:
            continue
        path = tse_dataset_dir / name
        if path.is_dir():
            return path
    return None


def _load_tse_wav_index(source_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    wav_scp = source_dir / "wav.scp"
    if wav_scp.exists():
        for utt, path in read_kaldi_format(wav_scp).items():
            index[str(utt)] = Path(path)
    for wav in source_dir.glob("*.wav"):
        index.setdefault(wav.stem, wav)
    return index


def _load_wav_file(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = soundfile.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio[0] if audio.shape[0] <= audio.shape[1] else audio[:, 0]
    return np.ascontiguousarray(audio.squeeze()), int(sr)


def _slice_wav_to_cut(audio: np.ndarray, sr: int, cut) -> np.ndarray:
    """Align a full-mix TSE wav with ``cut.load_audio()`` (cut-relative samples)."""
    start_i = int(round(float(cut.start) * sr))
    n = int(round(float(cut.duration) * sr))
    start_i = max(0, min(start_i, audio.shape[-1]))
    end_i = min(audio.shape[-1], start_i + max(0, n))
    return audio[start_i:end_i]


def load_tse_wav_for_cut(wav_index: dict, cut, sr: int) -> np.ndarray | None:
    """Load a mix-length TSE wav and slice it to the cut clock."""
    wav_path = wav_index.get(str(cut.id))
    if wav_path is None or not Path(wav_path).exists():
        return None
    audio, file_sr = _load_wav_file(Path(wav_path))
    if file_sr != sr and audio.size:
        import resampy
        audio = resampy.resample(audio, file_sr, sr)
    audio = _slice_wav_to_cut(audio, sr, cut)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def tse_turn_uses_s1(cut, speaker_id: str) -> bool:
    """True if this DiCoW speaker is SparseLibriMix s1 (``custom['anonymize']``)."""
    anon_local = cut_anonymize_speaker(cut, default=None)
    return anon_local is not None and str(speaker_id) == str(anon_local)


def build_tse_turn_audio_loader(
    tse_dataset_dir: Path,
    s1_subdir: str = "s1",
    residual_subdir: str = "s1_residual",
):
    """Return ``load_audio(cut, speaker_id)`` that yields TSE audio for the cut span."""
    tse_dataset_dir = Path(tse_dataset_dir)
    s1_dir = _resolve_tse_subdir(tse_dataset_dir, s1_subdir, _TSE_S1_ALIASES)
    residual_dir = _resolve_tse_subdir(tse_dataset_dir, residual_subdir, _TSE_RESIDUAL_ALIASES)
    if s1_dir is None and residual_dir is None:
        raise FileNotFoundError(
            f"No TSE s1/residual subdirs under {tse_dataset_dir} "
            f"(looked for {s1_subdir!r} / {residual_subdir!r})"
        )
    s1_index = _load_tse_wav_index(s1_dir) if s1_dir is not None else {}
    residual_index = _load_tse_wav_index(residual_dir) if residual_dir is not None else {}
    missing: set[tuple[str, str]] = set()
    stats = {"s1": 0, "residual": 0, "fallback_mix": 0}

    def load_audio(cut, speaker_id: str) -> tuple[np.ndarray, int]:
        cut_id = str(cut.id)
        use_s1 = tse_turn_uses_s1(cut, str(speaker_id))
        source = "s1" if use_s1 else "residual"
        wav_path = (s1_index if use_s1 else residual_index).get(cut_id)
        if wav_path is None or not Path(wav_path).exists():
            key = (cut_id, source)
            if key not in missing:
                missing.add(key)
                logger.warning(
                    f"TSE {source} wav missing for {cut_id}; "
                    "falling back to original mix for those turns"
                )
            stats["fallback_mix"] += 1
            return load_cut_audio(cut)
        audio, sr = _load_wav_file(Path(wav_path))
        audio = _slice_wav_to_cut(audio, sr, cut)
        stats[source] += 1
        return audio, sr

    load_audio.stats = stats  # type: ignore[attr-defined]
    return load_audio


def _replace_cut_recording(cut, rec: Recording):
    """Point ``cut`` at ``rec``."""
    if hasattr(cut, "recording"):
        kwargs = {"recording": rec}
        if hasattr(cut, "features"):
            kwargs["features"] = None
        return fastcopy(cut, **kwargs)
    custom = getattr(cut, "custom", None)
    if custom is not None:
        custom = dict(custom)
    return MonoCut(
        id=str(cut.id),
        start=0.0,
        duration=float(cut.duration),
        channel=0,
        supervisions=list(cut.supervisions),
        recording=rec,
        custom=custom,
    )


def attach_tse_s1_recordings(cutset: CutSet, s1_index: dict[str, Path]) -> CutSet:
    """Point each cut at its TSE s1 wav (same timeline as the mix)."""
    cuts = []
    n = 0
    for cut in cutset:
        wav = s1_index.get(str(cut.id))
        if wav is None or not Path(wav).exists():
            cuts.append(cut)
            continue
        rec_id = str(getattr(cut, "recording_id", cut.id))
        rec = Recording.from_file(str(Path(wav).resolve()), recording_id=rec_id)
        cuts.append(_replace_cut_recording(cut, rec))
        n += 1
    logger.info(f"Attached TSE s1 recordings to {n}/{len(cuts)} cuts")
    return CutSet.from_cuts(cuts)


def add_same_clock(anon: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Sample-wise sum; pad the shorter track with zeros (TSA paper step 3)."""
    a = np.asarray(anon, dtype=np.float32).reshape(-1)
    r = np.asarray(residual, dtype=np.float32).reshape(-1)
    n = max(a.size, r.size)
    out = np.zeros(n, dtype=np.float32)
    out[: a.size] += a
    out[: r.size] += r
    return out


def _fit_audio_to_length(wav: np.ndarray, n_target: int) -> np.ndarray:
    """Pad or crop to ``n_target`` samples. No phase-vocoder stretch."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if n_target <= 0:
        return np.zeros(0, dtype=np.float32)
    if wav.size == 0:
        return np.zeros(n_target, dtype=np.float32)
    if wav.size > n_target:
        return wav[:n_target]
    if wav.size < n_target:
        return np.pad(wav, (0, n_target - wav.size))
    return wav


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _scale_to_rms(sig: np.ndarray, target_rms: float) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32)
    r = _rms(sig)
    if r < 1e-8 or target_rms <= 0:
        return sig
    return sig * (target_rms / r)


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    mask_i = np.asarray(mask, dtype=np.int8).reshape(-1)
    d = np.diff(np.concatenate([[0], mask_i, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _crossfade_replaced_with_original(
    orig: np.ndarray,
    out: np.ndarray,
    replaced: np.ndarray,
    fade_len: int,
) -> np.ndarray:
    """Linear fade-in/out at the edges of replaced regions (TTS vs original mix)."""
    orig = np.asarray(orig, dtype=np.float32).reshape(-1)
    out = np.asarray(out, dtype=np.float32).copy().reshape(-1)
    if fade_len <= 0 or not np.any(replaced):
        return out
    for start, end in _contiguous_regions(replaced):
        length = end - start
        fade = min(int(fade_len), max(0, length // 2))
        if fade <= 0:
            continue
        w_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        sl = slice(start, start + fade)
        out[sl] = (1.0 - w_in) * orig[sl] + w_in * out[sl]
        w_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        sl = slice(end - fade, end)
        out[sl] = w_out * out[sl] + (1.0 - w_out) * orig[sl]
    return out


def _match_rms_envelope(sig: np.ndarray, ref: np.ndarray, hop: int = 256) -> np.ndarray:
    """Scale `sig` so its short-time RMS follows `ref` (original overlap energy)."""
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    n = min(sig.size, ref.size)
    if n == 0:
        return sig
    out = sig.copy()
    hop = max(1, int(hop))
    for i in range(0, n, hop):
        j = min(n, i + hop)
        r_sig = _rms(out[i:j])
        r_ref = _rms(ref[i:j])
        out[i:j] *= r_ref / r_sig
    return out


def _windows_to_mask(windows: list, sr: int, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=np.float32)
    for window in windows or []:
        start_i = min(max(0, int(round(float(window[0]) * sr))), n)
        end_i = min(max(start_i, int(round(float(window[1]) * sr))), n)
        if end_i > start_i:
            mask[start_i:end_i] = 1.0
    return mask


def remix_sum_tts(
    original_audio: np.ndarray,
    sr: int,
    tts_segments: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """Place each speaker TTS in its slot and sum. No original-mix exclusive keep."""
    n = np.asarray(original_audio, dtype=np.float32).reshape(-1).shape[0]
    canvas = np.zeros(n, dtype=np.float32)
    fit_log: list[dict] = []
    for seg in tts_segments:
        windows = seg.get("placement_windows") or [[seg["start"], seg["end"]]]
        sample_windows = []
        for window in windows:
            start_i = min(max(0, int(round(float(window[0]) * sr))), n)
            end_i = min(max(start_i, int(round(float(window[1]) * sr))), n)
            if end_i > start_i:
                sample_windows.append((start_i, end_i))
        n_slot = sum(e - s for s, e in sample_windows)
        if n_slot <= 0:
            continue
        wav = np.asarray(seg["audio"], dtype=np.float32).reshape(-1)
        tts_len = int(wav.size)
        if tts_len > n_slot:
            fit = "trim"
        elif tts_len < n_slot:
            fit = "pad"
        else:
            fit = "exact"
        fit_log.append({
            "utt": seg.get("utt"),
            "cut_id": seg.get("cut_id"),
            "speaker": str(seg.get("speaker", "")),
            "tts_samples": tts_len,
            "n_slot": int(n_slot),
            "tts_ms": round(1000.0 * tts_len / sr, 2),
            "n_slot_ms": round(1000.0 * n_slot / sr, 2),
            "delta_samples": tts_len - int(n_slot),
            "fit": fit,
        })
        fitted = _fit_audio_to_length(wav, n_slot)
        cursor = 0
        for start_i, end_i in sample_windows:
            length = end_i - start_i
            chunk = fitted[cursor:cursor + length]
            if chunk.size < length:
                chunk = np.pad(chunk, (0, length - chunk.size))
            canvas[start_i:end_i] += chunk[:length]
            cursor += length
    return canvas, fit_log


def remix_dialogue(
    original_audio: np.ndarray,
    sr: int,
    tts_segments: list[dict],
    fade_ms: float = 8.0,
    speaker_windows: dict | None = None,
    target_speaker_id: str = "0",
    min_exclusive_rms_sec: float = 0.5,
    min_exclusive_rms_ratio: float = 0.1,
    overlap_residual: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Replace exclusive target and overlap; keep exclusive non-target and silence.

    Exclusive target frames become anonymized TTS, scaled to the original
    exclusive-region RMS. Overlap is TTS(target) plus ``overlap_residual`` when
    that track is given (TSE s1_project / s2, not a second vocoder). Otherwise
    overlap is the sum of active speakers' TTS. Exclusive non-target and
    silence stay original.

    Exclusive RMS is treated as missing (overlap falls back to
    orig_overlap_rms / n_spk) if that region is shorter than
    ``min_exclusive_rms_sec`` or quieter than ``min_exclusive_rms_ratio``
    times the mix RMS — long diarization tails of near-silence would
    otherwise mute overlap TTS.
    """
    orig = np.array(original_audio, dtype=np.float32, copy=True).reshape(-1)
    n = orig.shape[0]
    fit_log: list[dict] = []
    if not tts_segments:
        return orig, fit_log

    # collect speakers' activity masks from tts_segments. speaker_windows is a dictionary of speaker ids and their placement windows. 
    # normally, speaker windows are provided (e.g. from diarization, the cutset supervisions), so this block is not needed.
    if speaker_windows is None:
        speaker_windows = {}
        for seg in tts_segments:
            spk = str(seg.get("speaker", "") or "")
            if not spk:
                continue
            windows = seg.get("placement_windows") or [[seg["start"], seg["end"]]]
            speaker_windows.setdefault(spk, []).extend(windows)
    speaker_windows = {str(spk): list(wins) for spk, wins in speaker_windows.items() if wins}
    if not speaker_windows:
        return orig, fit_log

    # create a tts-layout for each speaker. it's a recording of the orig length, where tts-segmenets of the speaker are placed in the right time slots.
    canvases: dict[str, np.ndarray] = {} # canvases[speaker] is a numpy array of shape (n, 1) that contains the tts audio of the speaker.
    for seg in tts_segments: # normally, one segment corresponds to one supervision (1 turn)
        spk = str(seg.get("speaker", "") or "")
        if not spk:
            continue
        windows = seg.get("placement_windows") or [[seg["start"], seg["end"]]]
        sample_windows = [] # same as windows, but in samples, not in seconds
        for window in windows:
            start_i = min(max(0, int(round(float(window[0]) * sr))), n)
            end_i = min(max(start_i, int(round(float(window[1]) * sr))), n)
            if end_i > start_i:
                sample_windows.append((start_i, end_i))
        n_slot = sum(e - s for s, e in sample_windows) # total length of the tts-layout for the speaker in samples
        if n_slot <= 0:
            continue
        wav = np.asarray(seg["audio"], dtype=np.float32).reshape(-1)
        tts_len = int(wav.size) # length of the tts segment in samples
        if tts_len > n_slot:
            fit = "trim" # trim the tts segment to the length of the tts-layout
        elif tts_len < n_slot:
            fit = "pad" # pad the tts segment to the length of the tts-layout
        else:
            fit = "exact" # the tts segment is exactly the length of the tts-layout
        fit_log.append({
            "utt": seg.get("utt"),
            "cut_id": seg.get("cut_id"),
            "speaker": spk,
            "tts_samples": tts_len,
            "n_slot": int(n_slot),
            "tts_ms": round(1000.0 * tts_len / sr, 2),
            "n_slot_ms": round(1000.0 * n_slot / sr, 2),
            "delta_samples": tts_len - int(n_slot),
            "fit": fit,
        })
        fitted = _fit_audio_to_length(wav, n_slot)
        if spk not in canvases:
            canvases[spk] = np.zeros(n, dtype=np.float32)
        cursor = 0
        for start_i, end_i in sample_windows:
            length = end_i - start_i
            chunk = fitted[cursor:cursor + length]
            if chunk.size < length:
                chunk = np.pad(chunk, (0, length - chunk.size))
            canvases[spk][start_i:end_i] = chunk[:length] # this cycle is basicly filling in the canvas with the tts audio of each speaker. tts segments are fitted to the length of the placement window and placed into the right time slot.
            cursor += length

    activity = {spk: _windows_to_mask(wins, sr, n) for spk, wins in speaker_windows.items()} # activity[speaker] is a numpy array of shape (n, 1) that contains 1 if the speaker is active at that time, 0 otherwise.
    n_active = np.zeros(n, dtype=np.float32) # how many speakers are active at this time step.
    for mask in activity.values():
        n_active += mask

    excl_rms = {}
    min_excl_samples = int(round(float(min_exclusive_rms_sec) * sr))
    mix_rms = _rms(orig)
    min_excl_rms = float(min_exclusive_rms_ratio) * mix_rms
    for spk, mask in activity.items():
        excl = (mask > 0) & (n_active == 1)
        n_excl = int(np.count_nonzero(excl))
        # Tiny or near-silent exclusive leftovers (short diarization tails,
        # false-positive "speech") give ~0 RMS and would mute overlap TTS.
        # Treat them as missing so overlap uses fallback = orig_overlap_rms / n_spk.
        if n_excl < max(1, min_excl_samples):
            excl_rms[spk] = 0.0
            continue
        raw = _rms(orig[excl])
        excl_rms[spk] = 0.0 if raw < min_excl_rms else raw

    # this block replaces the exclusive target regions with the tts-layout of the target speaker.
    # the replaced segments are scaled to the original exclusive-region RMS.
    out = orig.copy()
    replaced = np.zeros(n, dtype=bool)
    target_speaker_id = str(target_speaker_id)
    exclusive = n_active == 1
    for spk, mask in activity.items(): # for each speaker and their activity mask
        region = exclusive & (mask > 0) # region where "spk" is the only active speaker
        if not np.any(region) or spk not in canvases:
            continue
        if spk != target_speaker_id: # only anonymize target speaker. other exclusive regions remain unchanged.
            continue
        out[region] = _scale_to_rms(canvases[spk][region], excl_rms[spk])
        replaced[region] = True

    residual = None
    if overlap_residual is not None:
        residual = _fit_audio_to_length(
            np.asarray(overlap_residual, dtype=np.float32).reshape(-1), n
        )

    # Overlap: TTS(target)+residual if given, else sum of active TTS canvases.
    overlap = n_active >= 2
    if np.any(overlap):
        for start, end in _contiguous_regions(overlap):
            orig_seg = orig[start:end]
            active = [spk for spk, mask in activity.items() if np.any(mask[start:end] > 0)]
            if not active:
                continue
            if residual is not None:
                if target_speaker_id not in active or target_speaker_id not in canvases:
                    continue
                fallback = _rms(orig_seg) / 2.0
                t_rms = excl_rms.get(target_speaker_id, 0.0) or fallback
                out[start:end] = (
                    _scale_to_rms(canvases[target_speaker_id][start:end], t_rms)
                    + residual[start:end]
                )
                replaced[start:end] = True
                continue
            if any(spk not in canvases for spk in active):
                continue
            mixed = np.zeros(end - start, dtype=np.float32)
            fallback = _rms(orig_seg) / max(len(active), 1) # fallback rms value if the speaker's rms is 0
            for spk in active:
                target_rms = excl_rms[spk] if excl_rms[spk] > 0 else fallback
                mixed += _scale_to_rms(canvases[spk][start:end], target_rms)
            out[start:end] = mixed
            replaced[start:end] = True

    fade_len = int(round(float(fade_ms) * sr / 1000.0))
    out = _crossfade_replaced_with_original(orig, out, replaced, fade_len)
    return out, fit_log
