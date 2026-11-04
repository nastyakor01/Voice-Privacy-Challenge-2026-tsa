"""STTTS-Multi+DiCoW: RTTM clocks, concat TSE turns, remix_dialogue."""

from pathlib import Path

from ...modules.sttts_multi.dialogue.lhotse_io import (
    _TSE_RESIDUAL_ALIASES,
    _load_tse_wav_index,
    _resolve_tse_subdir,
    load_tse_wav_for_cut,
    resolve_tse_dataset_dir,
    write_cutset_rttm,
)
from ..sttts_multi_dicow.sttts_multi_dicow_pipeline import STTTSMultiDiCoWPipeline
from utils import setup_logger

logger = setup_logger(__name__)


class STTTSMultiDiCoWConcatPipeline(STTTSMultiDiCoWPipeline):
    """DiCoW as text only; input supervisions are the only time structure.

    * No Whisper timestamps.
    * One TTS per speaker: aligner sees TSE s1 / residual concatenated on
      that speaker's turns; remix splits the wav back into those slots.
    * ``remix_dialogue``: exclusive other + silence stay original; exclusive
      target is TTS; overlap is TTS(target) + TSE residual (not a second TTS).
    * Input RTTM is copied next to the remixed wavs.
    """

    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        asr = config.setdefault("modules", {}).setdefault("asr", {})
        asr["return_timestamps"] = False
        asr["segment_layout"] = "concat"
        super().__init__(config=config, force_compute=force_compute, devices=devices)
        self._residual_index: dict[str, Path] = {}

    def _remix_tag(self) -> str:
        if (self.modules_config.get("prosody") or {}).get("tse"):
            return "remix_concat_tse"
        return "remix_concat"

    def _prepare_cutset(self, cutset, dataset_name: str):
        self._residual_index = {}
        prosody_cfg = self.modules_config.get("prosody") or {}
        tse_root = Path(prosody_cfg.get("tse_dir") or (self.intermediate_dir / "tse"))
        tse_ds = resolve_tse_dataset_dir(tse_root, dataset_name)
        if tse_ds is None:
            logger.warning(
                f"No TSE dir for {dataset_name} under {tse_root}; "
                "overlap remix stays TTS+TTS"
            )
            return cutset
        res_dir = _resolve_tse_subdir(
            tse_ds,
            str(prosody_cfg.get("tse_residual_subdir", "s1_project")),
            _TSE_RESIDUAL_ALIASES,
        )
        if res_dir is None:
            logger.warning(
                f"No residual subdir under {tse_ds}; overlap remix stays TTS+TTS"
            )
            return cutset
        self._residual_index = _load_tse_wav_index(res_dir)
        logger.info(
            f"Overlap residual index from {res_dir}: {len(self._residual_index)} wavs"
        )
        return cutset

    def _overlap_residual_audio(self, cut, sr: int):
        if not self._residual_index:
            return None
        residual = load_tse_wav_for_cut(self._residual_index, cut, sr)
        if residual is None:
            logger.warning(
                f"TSE residual missing for {cut.id}; overlap remix falls back to TTS+TTS"
            )
        return residual

    def _write_preserved_diarization(self, cutset, remixed_dir: Path, output_path: Path):
        for dest in (remixed_dir / "diarization.rttm", output_path / "diarization.rttm"):
            write_cutset_rttm(cutset, dest)
            logger.info(f"Wrote input supervisions to {dest}")
