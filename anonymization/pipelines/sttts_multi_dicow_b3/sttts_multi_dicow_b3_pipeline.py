"""TSE s1 → STTTS-Multi+DiCoW → add residual (paper 2510.09307, STTTS instead of ASRBN)."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import soundfile

from ...modules.sttts_multi.dialogue.lhotse_io import (
    _TSE_RESIDUAL_ALIASES,
    _TSE_S1_ALIASES,
    _fit_audio_to_length,
    _load_tse_wav_index,
    _load_wav_file,
    _resolve_tse_subdir,
    add_same_clock,
    attach_tse_s1_recordings,
    cut_anonymize_speaker,
    load_cut_audio,
    resolve_tse_dataset_dir,
)
from ..sttts_multi_dicow.sttts_multi_dicow_pipeline import STTTSMultiDiCoWPipeline
from utils import setup_logger

logger = setup_logger(__name__)


class STTTSMultiDiCoWB3Pipeline(STTTSMultiDiCoWPipeline):
    """Same STTTS stack as ``sttts_multi_dicow``, TSA remix like ``multi_asrbn``.

    * DiCoW without timestamps; STNO is all-target (no supervision clock).
    * One TTS over the entire TSE s1; aligner sees the full s1 wav.
    * ECAPA from ``enroll.scp`` / TSE support enrolls if ``use_tse_support_enrolls``
      (default true); else exclusive/turns, or the full cut when ``_embed_full_cut``.
    * Remix is ``fit(TTS) + residual`` (``s1_project``), not ``remix_dialogue``.
    """

    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        asr = config.setdefault("modules", {}).setdefault("asr", {})
        asr["return_timestamps"] = False
        asr["segment_layout"] = "full"
        asr["decode_anonymize_only"] = True
        asr["stno_layout"] = "all_target"
        super().__init__(config=config, force_compute=force_compute, devices=devices)
        self._residual_index: dict[str, Path] = {}
        self._embed_cutset = None

    def _remix_tag(self) -> str:
        return "remix_b3"

    def _embed_full_cut(self) -> bool:
        return True

    def _prepare_cutset(self, cutset, dataset_name: str):
        self._embed_cutset = cutset
        self._residual_index = {}
        prosody_cfg = self.modules_config.get("prosody") or {}
        tse_root = Path(prosody_cfg.get("tse_dir") or (self.intermediate_dir / "tse"))
        tse_ds = resolve_tse_dataset_dir(tse_root, dataset_name)
        if tse_ds is None:
            logger.warning(
                f"No TSE dir for {dataset_name} under {tse_root}; "
                "STTTS on original audio, no residual add"
            )
            return cutset
        s1_dir = _resolve_tse_subdir(tse_ds, str(prosody_cfg.get("tse_s1_subdir", "s1")), _TSE_S1_ALIASES)
        res_dir = _resolve_tse_subdir(
            tse_ds,
            str(prosody_cfg.get("tse_residual_subdir", "s1_project")),
            _TSE_RESIDUAL_ALIASES,
        )
        if s1_dir is None:
            logger.warning(f"No TSE s1 under {tse_ds}; STTTS on original audio")
            return cutset
        cutset = attach_tse_s1_recordings(cutset, _load_tse_wav_index(s1_dir))
        if res_dir is not None:
            self._residual_index = _load_tse_wav_index(res_dir)
            logger.info(f"Residual index from {res_dir}: {len(self._residual_index)} wavs")
        else:
            logger.warning(f"No residual subdir under {tse_ds}; will write TTS only")
        return cutset

    def _remix_dataset(self, cutset, segments, tts_wav_scp, remixed_dir: Path):
        segs_by_cut = {}
        for seg in segments:
            segs_by_cut.setdefault(str(seg["cut_id"]), []).append(seg)

        mix_wav_scp = {}
        fit_log_path = remixed_dir / "tts_slot_fit.jsonl"
        fit_records = []
        skipped = 0
        for cut in cutset:
            orig_audio, sr = load_cut_audio(cut)
            out_path = remixed_dir / f"{cut.id}.wav"
            target = cut_anonymize_speaker(cut, default=self.target_speaker_id)
            tts_audio = None
            for seg in segs_by_cut.get(str(cut.id), []):
                if target is not None and str(seg.get("speaker")) != str(target):
                    continue
                wav_ref = tts_wav_scp.get(seg["utt"])
                if wav_ref is None:
                    continue
                wav, tts_sr = soundfile.read(wav_ref)
                if tts_sr != sr and wav.size:
                    import resampy
                    wav = resampy.resample(wav, tts_sr, sr)
                tts_audio = wav
                break
            if tts_audio is None or target is None:
                remixed = orig_audio
                skipped += 1
            else:
                fitted = _fit_audio_to_length(tts_audio, orig_audio.size)
                tts_len = int(np.asarray(tts_audio).reshape(-1).size)
                n_slot = int(orig_audio.size)
                if tts_len > n_slot:
                    fit = "trim"
                elif tts_len < n_slot:
                    fit = "pad"
                else:
                    fit = "exact"
                fit_records.append({
                    "utt": str(cut.id),
                    "cut_id": str(cut.id),
                    "speaker": str(target),
                    "tts_samples": tts_len,
                    "n_slot": n_slot,
                    "fit": fit,
                })
                res_path = self._residual_index.get(str(cut.id))
                if res_path is not None and Path(res_path).exists():
                    residual, res_sr = _load_wav_file(Path(res_path))
                    if res_sr != sr and residual.size:
                        import resampy
                        residual = resampy.resample(residual, res_sr, sr)
                    remixed = add_same_clock(fitted, residual)
                else:
                    remixed = fitted
            soundfile.write(str(out_path), remixed, sr)
            mix_wav_scp[str(cut.id)] = str(out_path.resolve())
        if skipped:
            logger.warning(f"Wrote original/TSE audio for {skipped} cuts without target TTS")
        with fit_log_path.open("w", encoding="utf-8") as f:
            for row in fit_records:
                f.write(json.dumps(row) + "\n")
        logger.info(f"Wrote TTS vs s1 lengths to {fit_log_path}")
        return mix_wav_scp
