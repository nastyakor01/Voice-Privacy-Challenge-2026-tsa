from collections import defaultdict
from pathlib import Path
import json
import time
import soundfile
import torch

from .. import Pipeline, get_anon_level_from_config
from utils import copy_data_dir, save_kaldi_format, setup_logger

from ...modules.sttts_multi.tts import SpeechSynthesis
from ...modules.sttts_multi.text import Text
from ...modules.sttts_multi.text.recognition.dicow import DiCoWASR
from ...modules.sttts_multi.prosody import ProsodyExtraction, ProsodyAnonymization
from ...modules.sttts_multi.speaker_embeddings.speaker_extraction import SpeakerExtraction
from ...modules.sttts_multi.speaker_embeddings.speaker_anonymization import SpeakerAnonymization
from ...modules.sttts_multi.speaker_embeddings.speaker_embeddings import SpeakerEmbeddings
from ...modules.sttts_multi.dialogue.lhotse_io import (
    load_cutset,
    write_mix_kaldi_dir,
    write_speaker_segment_kaldi_dir,
    write_turn_kaldi_dir,
    mix_speaker_embedding_utt,
    load_cut_audio,
    remix_dialogue,
    cut_speakers,
    cut_anonymize_speaker,
    cut_speaker_windows,
    resolve_tse_dataset_dir,
    build_tse_turn_audio_loader,
)
import typing

logger = setup_logger(__name__)


class STTTSMultiDiCoWPipeline(Pipeline):
    """STTTS-Multi anonymization that transcribes dialogue speakers with DiCoW.

    Differences vs STTTSMultiPipeline:
      * ASR is BUT-FIT/DiCoW (target-speaker Whisper), not openai/whisper-large-v3
      * diarization comes from a saved lhotse cutset (source of STNO masks)
      * per-cut ``custom['anonymize']`` (mix-local speaker id) is GAN-anonymized;
        yaml ``target_speaker_id`` is the fallback if custom is missing
      * exclusive non-target and silence stay original; overlap is mixed from
        anonymized target TTS and non-target TTS (original embeddings, no GAN)
    """

    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        self.total_start_time = time.time()
        self.config = config
        model_dir = Path(config.get("models_dir", "models"))
        vectors_dir = Path(config.get("vectors_dir", "original_speaker_embeddings"))
        self.results_dir = Path(config.get("results_dir", "results"))
        self.intermediate_dir = Path(config.get("intermediate_dir", "exp/intermediate_dir_stts"))
        self.data_dir = Path(config["data_dir"]) if "data_dir" in config else None
        self.anon_suffix = config.get("anon_suffix", "_sttts_dicow")
        save_intermediate = config.get("save_intermediate", True)
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute

        modules_config = config["modules"]
        self.modules_config = modules_config
        asr_settings = modules_config["asr"]
        self.target_speaker_id = str(asr_settings.get("target_speaker_id", "0"))
        self.asr_language = asr_settings.get("language", "en")

        self.dicow = DiCoWASR(device=devices[0], **asr_settings)
        asr_results = asr_settings.get("results_path") or asr_settings.get("results_dir")
        self.asr_results_dir = Path(asr_results) if asr_results else self.intermediate_dir / "transcription" / "dicow"

        self.speaker_extraction = {}
        self.speaker_anonymization = {}
        for level in ["utt", "spk"]:
            modules_config["speaker_embeddings"]["emb_level"] = level
            self.speaker_extraction[level] = SpeakerExtraction(
                devices=devices,
                save_intermediate=save_intermediate,
                settings=modules_config["speaker_embeddings"],
            )
            self.speaker_anonymization[level] = SpeakerAnonymization(
                vectors_dir=vectors_dir,
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["speaker_embeddings"],
                force_compute=force_compute,
            )

        if "prosody" in modules_config:
            self.prosody_extraction = ProsodyExtraction(
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["prosody"],
            )
            if "anonymizer_type" in modules_config["prosody"]:
                self.prosody_anonymization = ProsodyAnonymization(
                    save_intermediate=save_intermediate,
                    settings=modules_config["prosody"],
                    force_compute=force_compute,
                )
            else:
                self.prosody_anonymization = None
        else:
            self.prosody_extraction = None
            self.prosody_anonymization = None

        self.speech_synthesis = SpeechSynthesis(
            devices=devices,
            settings=modules_config["tts"],
            model_dir=model_dir,
            save_output=config.get("save_output", True),
            force_compute=force_compute,
        )

    def _resolve_cutset_path(self, dataset_name: str) -> Path:
        asr_cfg = self.modules_config["asr"]
        mapping = asr_cfg.get("lhotse_cutsets") or {}
        if dataset_name in mapping:
            return Path(mapping[dataset_name])
        for entry in self.config.get("datasets", []):
            name = entry.get("data") or entry.get("name")
            if name == dataset_name and entry.get("lhotse_cutset"):
                return Path(entry["lhotse_cutset"])
        if asr_cfg.get("lhotse_cutset"):
            return Path(asr_cfg["lhotse_cutset"])
        raise FileNotFoundError(
            f"No lhotse cutset configured for dataset {dataset_name}. "
            f"Set datasets[].lhotse_cutset or modules.asr.lhotse_cutset(s)."
        )

    def _asr_cache_covers_speakers(self, out_dir: Path, cutset) -> bool:
        coverage_path = out_dir / "decoded_speakers.jsonl"
        if coverage_path.exists():
            decoded = defaultdict(set)
            for line in coverage_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                decoded[str(row["cut_id"])].add(str(row["speaker"]))
        else:
            decoded = None
        for cut in cutset:
            target = cut_anonymize_speaker(cut, default=self.target_speaker_id)
            if target is None:
                continue
            need = set(cut_speakers(cut))
            if decoded is None:
                return False
            if not need.issubset(decoded.get(str(cut.id), set())):
                return False
        return True

    def _load_or_run_asr(self, cutset, dataset_name: str):
        out_dir = self.asr_results_dir / dataset_name
        segments_path = out_dir / "segments.jsonl"
        texts = Text(is_phones=False)
        force = self.force_compute or self.modules_config["asr"].get("force_compute_recognition", False)
        if segments_path.exists() and (out_dir / "text").exists() and not force:
            logger.info(f"Load existing DiCoW transcripts from {out_dir}")
            segments = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines() if line]
            if segments and self._asr_cache_covers_speakers(out_dir, cutset):
                texts.load_text(in_dir=out_dir)
                return texts, segments
            logger.info("Cached DiCoW transcripts are empty or incomplete; re-running ASR")

        out_dir.mkdir(parents=True, exist_ok=True)
        texts, segments = self.dicow.recognize_cutset(
            cutset,
            text_instance=texts,
            out_dir=out_dir,
            save_intermediate=self.save_intermediate,
        )
        if self.save_intermediate:
            texts.save_text(out_dir=out_dir)
            with segments_path.open("w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg) + "\n")
        return texts, segments

    def _cut_targets(self, cutset) -> dict:
        """cut.id -> speaker id from ``custom['anonymize']`` (yaml fallback)."""
        targets = {}
        n_custom = n_default = n_skip = 0
        counts = defaultdict(int)
        for cut in cutset:
            custom = getattr(cut, "custom", None) or {}
            raw = custom.get("anonymize")
            target = cut_anonymize_speaker(cut, default=self.target_speaker_id)
            if target is None:
                n_skip += 1
                continue
            targets[str(cut.id)] = target
            counts[target] += 1
            if raw is None or raw == "":
                n_default += 1
            else:
                n_custom += 1
        logger.info(
            "Per-cut anonymize speakers: "
            f"{n_custom} from custom, {n_default} yaml fallback "
            f"({self.target_speaker_id}), skipped {n_skip}; counts={dict(counts)}"
        )
        return targets

    def _prepare_tts_embeddings(self, utt_embeddings, segments, dataset_name, anon_level, cut_targets: dict):
        """GAN-anonymize only the per-cut target speaker; keep original embeddings for others.

        One ECAPA per (mix, speaker) on concatenated exclusive speech, or on all
        turns if exclusive speech in the mix is shorter than 2s. Target: that
        vector through GAN once, reused on every target turn. Non-target: same
        mix-level vector, no GAN. ``cut_targets`` maps cut.id -> speaker id from
        ``cut.custom['anonymize']``.
        """
        groups = defaultdict(list)  # (cut_id, speaker) -> [asr utt, ...]
        for seg in segments:
            speaker = seg.get("speaker")
            if speaker is None:
                continue
            groups[(str(seg["cut_id"]), str(speaker))].append(seg["utt"])

        def _mix_meta(cut_id, speaker):
            mix_utt = mix_speaker_embedding_utt(cut_id, speaker)
            if mix_utt not in utt_embeddings.identifiers2idx:
                return None
            idx = utt_embeddings.identifiers2idx[mix_utt]
            return (
                utt_embeddings.get_embedding_for_identifier(mix_utt),
                utt_embeddings.original_speakers[int(idx)],
                utt_embeddings.genders[int(idx)],
            )

        target_groups = {}
        other_groups = {}
        for (cut_id, speaker), uids in groups.items():
            target = cut_targets.get(cut_id)
            if target is not None and speaker == target:
                target_groups[cut_id] = uids
            else:
                other_groups[(cut_id, speaker)] = uids

        # TTS looks up embeddings at ASR utt-level, so broadcast the mix vector.
        vec_by_utt = {}
        meta_by_utt = {}
        if target_groups:
            mix_ids, mix_vecs, mix_spks, mix_genders = [], [], [], []
            ready = {}
            for cut_id, uids in target_groups.items():
                meta = _mix_meta(cut_id, cut_targets[cut_id])
                if meta is None:
                    continue
                vec, spk, gender = meta
                mix_ids.append(cut_id)
                mix_vecs.append(vec)
                mix_spks.append(spk)
                mix_genders.append(gender)
                ready[cut_id] = (uids, spk, gender)
            if mix_ids:
                mix_embs = SpeakerEmbeddings(
                    vec_type=utt_embeddings.vec_type, emb_level="utt", device=utt_embeddings.device
                )
                mix_embs.set_vectors(mix_ids, torch.stack(mix_vecs, dim=0), mix_spks, mix_genders)

                anon_key = anon_level if anon_level in self.speaker_anonymization else "utt"
                if self.speaker_anonymization and anon_key in self.speaker_anonymization:
                    start_time = time.time()
                    anon_mix = self.speaker_anonymization[anon_key].anonymize_embeddings(
                        speaker_embeddings=mix_embs, dataset_name=f"{dataset_name}_target_mix"
                    )
                    logger.info("--- Speaker anonymization (one GAN voice per mix) time: "
                                f"{(float(time.time() - start_time) / 60)} min ---")
                else:
                    anon_mix = mix_embs

                for cut_id, (uids, spk, gender) in ready.items():
                    anon_vec = anon_mix.get_embedding_for_identifier(cut_id)
                    for uid in uids:
                        vec_by_utt[uid] = anon_vec
                        meta_by_utt[uid] = (spk, gender)

        for (cut_id, speaker), uids in other_groups.items():
            meta = _mix_meta(cut_id, speaker)
            if meta is None:
                continue
            vec, spk, gender = meta
            for uid in uids:
                vec_by_utt[uid] = vec
                meta_by_utt[uid] = (spk, gender)

        identifiers, vectors, speakers, genders = [], [], [], []
        for seg in segments:
            uid = seg["utt"]
            if uid not in vec_by_utt:
                continue
            spk, gender = meta_by_utt[uid]
            identifiers.append(uid)
            vectors.append(vec_by_utt[uid].reshape(-1))
            speakers.append(spk)
            genders.append(gender)

        merged = SpeakerEmbeddings(vec_type=utt_embeddings.vec_type, emb_level="utt", device=utt_embeddings.device)
        if identifiers:
            merged.set_vectors(identifiers, torch.stack(vectors, dim=0), speakers, genders)
        return merged

    def run_anonymization_pipeline(self, datasets: typing.Dict[str, Path]):
        anon_wav_scps = {}

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            logger.info(f"{i + 1}/{len(datasets)}: Processing {dataset_name} "
                        f"(per-cut custom['anonymize'], yaml fallback {self.target_speaker_id})...")
            cutset_path = self._resolve_cutset_path(dataset_name)
            logger.info(f"Lhotse cutset: {cutset_path}")
            cutset = load_cutset(cutset_path)
            cut_targets = self._cut_targets(cutset)

            mix_kaldi_dir = Path(dataset_path)
            write_mix_kaldi_dir(cutset, mix_kaldi_dir)

            start_time = time.time()
            texts, segments = self._load_or_run_asr(cutset, dataset_name)
            logger.info("--- Speech recognition time: %f min ---" % (float(time.time() - start_time) / 60))

            speaker_kaldi_dir = self.intermediate_dir / "all_speaker_kaldi" / dataset_name
            speaker_wav_dir = speaker_kaldi_dir / "wav"
            if segments:
                write_speaker_segment_kaldi_dir(
                    cutset=cutset,
                    segments=segments,
                    out_dir=speaker_kaldi_dir,
                    wav_dir=speaker_wav_dir,
                )
            else:
                logger.warning("No DiCoW segments; original mixtures will be copied.")

            start_time = time.time()
            anon_level = get_anon_level_from_config(self.modules_config["speaker_embeddings"], dataset_name)
            extract_dataset_name = f"{dataset_name}_allspk"
            if segments:
                spk_embeddings = self.speaker_extraction["utt"].extract_speakers(
                    dataset_path=speaker_kaldi_dir, dataset_name=extract_dataset_name
                )
            else:
                spk_embeddings = None
            logger.info("--- Speaker extraction (utt) time: "
                        f"{(float(time.time() - start_time) / 60)} min ---")

            if self.prosody_extraction and segments:
                start_time = time.time()
                turn_kaldi_dir = self.intermediate_dir / "turn_kaldi" / dataset_name
                write_turn_kaldi_dir(
                    cutset=cutset,
                    segments=segments,
                    out_dir=turn_kaldi_dir,
                    wav_dir=turn_kaldi_dir / "wav",
                )
                prosody_kaldi_dir = turn_kaldi_dir
                prosody_dataset_name = dataset_name
                prosody_cfg = self.modules_config.get("prosody") or {}
                if prosody_cfg.get("tse"):
                    tse_root = Path(prosody_cfg.get("tse_dir") or (self.intermediate_dir / "tse"))
                    tse_ds = resolve_tse_dataset_dir(tse_root, dataset_name)
                    if tse_ds is None:
                        logger.warning(
                            f"prosody.tse=true but no TSE dir for {dataset_name} under {tse_root}; "
                            "using original-mix turn_kaldi crops for prosody"
                        )
                    else:
                        turn_tse_kaldi_dir = self.intermediate_dir / "turn_tse_kaldi" / dataset_name
                        tse_loader = build_tse_turn_audio_loader(
                            tse_dataset_dir=tse_ds,
                            s1_subdir=str(prosody_cfg.get("tse_s1_subdir", "s1")),
                            residual_subdir=str(prosody_cfg.get("tse_residual_subdir", "s1_residual")),
                        )
                        write_turn_kaldi_dir(
                            cutset=cutset,
                            segments=segments,
                            out_dir=turn_tse_kaldi_dir,
                            wav_dir=turn_tse_kaldi_dir / "wav",
                            load_audio=tse_loader,
                        )
                        stats = getattr(tse_loader, "stats", {})
                        logger.info(
                            f"Wrote TSE turn crops to {turn_tse_kaldi_dir} from {tse_ds} "
                            f"(s1={stats.get('s1', 0)}, residual={stats.get('residual', 0)}, "
                            f"fallback_mix={stats.get('fallback_mix', 0)})"
                        )
                        prosody_kaldi_dir = turn_tse_kaldi_dir
                        # Separate cache so mix-crop duration.pt is not reused.
                        prosody_dataset_name = f"{dataset_name}_tse"
                prosody = self.prosody_extraction.extract_prosody(
                    dataset_path=prosody_kaldi_dir,
                    dataset_name=prosody_dataset_name,
                    texts=texts,
                )
                logger.info(f"--- Prosody extraction time: {(float(time.time() - start_time) / 60)} min ---")
            else:
                prosody = None

            if spk_embeddings is not None:
                anon_embeddings = self._prepare_tts_embeddings(
                    utt_embeddings=spk_embeddings,
                    segments=segments,
                    dataset_name=dataset_name,
                    anon_level=anon_level,
                    cut_targets=cut_targets,
                )
            else:
                anon_embeddings = None

            if self.prosody_anonymization and prosody is not None:
                start_time = time.time()
                anon_prosody = self.prosody_anonymization.anonymize_prosody(prosody=prosody, dataset_name=dataset_name)
                logger.info("--- Prosody anonymization time: %f min ---" % (float(time.time() - start_time) / 60))
            else:
                anon_prosody = prosody

            tts_wav_scp = {}
            if segments and anon_embeddings is not None and len(texts) > 0:
                start_time = time.time()
                tts_wav_scp = self.speech_synthesis.synthesize_speech(
                    dataset_name=dataset_name,
                    texts=texts,
                    speaker_embeddings=anon_embeddings,
                    prosody=anon_prosody,
                    emb_level="utt",
                )
                logger.info("--- Synthesis time: %f min ---" % (float(time.time() - start_time) / 60))

            start_time = time.time()
            remix_tag = "remix_tse" if (self.modules_config.get("prosody") or {}).get("tse") else "remix"
            remixed_dir = self.modules_config["tts"]["results_path"] / f"{dataset_name}_{remix_tag}"
            remixed_dir.mkdir(parents=True, exist_ok=True)
            mix_wav_scp = self._remix_dataset(cutset, segments, tts_wav_scp, remixed_dir)
            logger.info("--- Remix time: %f min ---" % (float(time.time() - start_time) / 60))

            output_path = Path(str(mix_kaldi_dir) + self.anon_suffix)
            copy_data_dir(mix_kaldi_dir, output_path)
            save_kaldi_format(mix_wav_scp, output_path / "wav.scp")
            anon_wav_scps[dataset_name] = mix_wav_scp
            logger.info(f"Wrote remixed mixtures to {output_path}")

        logger.info("--- Total computation time: %f min ---" % (float(time.time() - self.total_start_time) / 60))
        return anon_wav_scps

    def _remix_dataset(self, cutset, segments, tts_wav_scp, remixed_dir: Path):
        segs_by_cut = {}
        # for each cut_id (e.g. mix_0000001) get the list of segments for this cut
        for seg in segments:
            segs_by_cut.setdefault(str(seg["cut_id"]), []).append(seg)

        mix_wav_scp = {}
        fit_log_path = remixed_dir / "tts_slot_fit.jsonl"
        fit_records = []
        for cut in cutset:
            orig_audio, sr = load_cut_audio(cut)
            out_path = remixed_dir / f"{cut.id}.wav"
            cut_segs = segs_by_cut.get(str(cut.id), []) # take all segments for this cut
            tts_segments = []
            for seg in cut_segs:
                wav_ref = tts_wav_scp.get(seg["utt"])
                if wav_ref is None:
                    continue
                tts_audio, tts_sr = soundfile.read(wav_ref)
                if tts_sr != sr and tts_audio.size:
                    import resampy
                    tts_audio = resampy.resample(tts_audio, tts_sr, sr)
                tts_segments.append({
                    "utt": seg["utt"],
                    "cut_id": str(cut.id),
                    "start": seg["start"],
                    "end": seg["end"],
                    "audio": tts_audio,
                    "speaker": str(seg.get("speaker", self.target_speaker_id)),
                    "placement_windows": seg.get("placement_windows") or [[seg["start"], seg["end"]]],
                })

            target = cut_anonymize_speaker(cut, default=self.target_speaker_id)
            if tts_segments and target is not None:
                remixed, cut_fit = remix_dialogue(
                    orig_audio,
                    sr,
                    tts_segments,
                    speaker_windows=cut_speaker_windows(cut),
                    target_speaker_id=target,
                )
                fit_records.extend(cut_fit)
            else:
                remixed = orig_audio
            soundfile.write(str(out_path), remixed, sr)
            mix_wav_scp[str(cut.id)] = str(out_path.resolve())
        with fit_log_path.open("w", encoding="utf-8") as f:
            for row in fit_records:
                f.write(json.dumps(row) + "\n")
        logger.info(f"Wrote TTS vs slot lengths to {fit_log_path}")
        return mix_wav_scp
