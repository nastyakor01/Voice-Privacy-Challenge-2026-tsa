from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .dicow_model import DiCoWForConditionalGeneration

from ..text import Text
from ...dialogue.lhotse_io import (
    build_stno_mask,
    build_all_target_stno_mask,
    cut_has_speaker,
    cut_speakers,
    cut_anonymize_speaker,
    hyp_to_target_turn_segments,
    hyp_to_concat_segment,
    hyp_to_span_segment,
    load_cut_audio,
    target_supervision_windows,
    cut_audio_window,
)
from utils import setup_logger

logger = setup_logger(__name__)


def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower_cased_token = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
        else:
            lower_cased_token = token[0].lower() + token[1:]
        if lower_cased_token != token:
            lower_index = vocab.get(lower_cased_token, None)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index


class DiCoWASR:
    """DiCoW ASR for every speaker on a mix, conditioned on lhotse diarization."""

    def __init__(
        self,
        device,
        model_name="BUT-FIT/DiCoW_v3_2",
        model_path=None,
        target_speaker_id="0",
        language="en",
        batch_size=4,
        utt_start_token="",
        utt_end_token="",
        decoding_ctc_weight=0.0,
        num_beams=1,
        return_timestamps=True,
        segment_layout="turns",
        decode_anonymize_only=False,
        stno_layout="supervisions",
        **kwargs,
    ):
        self.device = device
        self.model_name = model_name
        self.model_path = model_path
        self.target_speaker_id = str(target_speaker_id)
        self.language = language or "en"
        self.batch_size = max(1, int(batch_size))
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token
        self.return_timestamps = return_timestamps
        self.segment_layout = str(segment_layout or "turns")
        self.decode_anonymize_only = bool(decode_anonymize_only)
        self.stno_layout = str(stno_layout or "supervisions")
        self.num_beams = num_beams
        self.output = "text"

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        cache_dir = str(model_path) if model_path else None
        self.torch_dtype = torch_dtype

        logger.info(f"Loading DiCoW ASR from {model_name} (local 4.55-compatible classes)")
        # Do not use HF trust_remote_code: hub files import WHISPER_ATTENTION_CLASSES (transformers 4.42).
        self.model = DiCoWForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        create_lower_uppercase_mapping(self.tokenizer)
        if hasattr(self.model, "set_tokenizer"):
            self.model.set_tokenizer(self.tokenizer)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.ctc_weight = decoding_ctc_weight

        self.n_samples = int(getattr(self.feature_extractor, "n_samples", 30 * 16000))
        self.hop_length = int(getattr(self.feature_extractor, "hop_length", 160))

    def _extract_features(self, audio, sr):
        batch = self.feature_extractor(
            audio,
            return_tensors="pt",
            sampling_rate=sr,
            return_attention_mask=True,
            truncation=False,
            padding="longest",
            pad_to_multiple_of=self.n_samples,
        )
        input_features = batch["input_features"][0]
        attention_mask = batch["attention_mask"][0]
        return input_features, attention_mask

    def _prepare_features(self, cut):
        audio, sr = load_cut_audio(cut)
        input_features, attention_mask = self._extract_features(audio, sr)
        n_feat_frames = int(input_features.shape[-1])
        n_enc_frames = max(1, n_feat_frames // 2)
        n_samples_padded = n_feat_frames * self.hop_length
        return {
            "cut": cut,
            "input_features": input_features,
            "attention_mask": attention_mask,
            "n_enc_frames": n_enc_frames,
            "n_samples_padded": n_samples_padded,
        }

    def _items_for_cut(self, cut, speakers):
        base = self._prepare_features(cut)
        items = []
        for spk in speakers:
            if self.stno_layout == "all_target":
                stno = build_all_target_stno_mask(base["n_enc_frames"])
            else:
                if not cut_has_speaker(cut, spk):
                    continue
                stno = build_stno_mask(
                    cut,
                    speaker_id=spk,
                    n_enc_frames=base["n_enc_frames"],
                    n_samples_padded=base["n_samples_padded"],
                )
            items.append({
                "cut": cut,
                "speaker": str(spk),
                "input_features": base["input_features"],
                "attention_mask": base["attention_mask"],
                "stno_mask": torch.from_numpy(stno),
            })
        return items

    def _collate(self, items):
        max_t = max(x["input_features"].shape[-1] for x in items)
        max_stno = max_t // 2
        feats, attn, stno = [], [], []
        for x in items:
            feat = x["input_features"]
            mask = x["attention_mask"]
            st = x["stno_mask"]
            if feat.shape[-1] < max_t:
                pad_t = max_t - feat.shape[-1]
                feat = torch.nn.functional.pad(feat, (0, pad_t))
                mask = torch.nn.functional.pad(mask, (0, pad_t))
            if st.shape[-1] < max_stno:
                pad_s = max_stno - st.shape[-1]
                st = torch.nn.functional.pad(st, (0, pad_s))
                st[0, st.shape[-1] - pad_s:] = 1.0
            feats.append(feat)
            attn.append(mask)
            stno.append(st)
        return torch.stack(feats, dim=0), torch.stack(attn, dim=0), torch.stack(stno, dim=0)

    def _generate_batch(self, items):
        feats, attn, stno = self._collate(items)
        feats = feats.to(device=self.device, dtype=self.torch_dtype)
        attn = attn.to(self.device)
        stno = stno.to(device=self.device, dtype=self.torch_dtype)
        with torch.inference_mode():
            output = self.model.generate(
                input_features=feats,
                attention_mask=attn,
                stno_mask=stno,
                language=self.language,
                task="transcribe",
                return_timestamps=self.return_timestamps,
                num_beams=self.num_beams,
                condition_on_prev_tokens=False,
            )
        if isinstance(output, dict) and "sequences" in output:
            tokens = output["sequences"]
        else:
            tokens = output
        return self.tokenizer.batch_decode(tokens, decode_with_timestamps=True, skip_special_tokens=True)

    def _segments_from_hyp(self, cut, hyp_text, speaker_id=None):
        speaker_id = str(speaker_id or self.target_speaker_id)
        windows = target_supervision_windows(cut, speaker_id)
        if self.segment_layout == "full":
            window = cut_audio_window(cut)
            parsed = hyp_to_span_segment(hyp_text, [window] if window else [])
        elif self.segment_layout == "span":
            parsed = hyp_to_span_segment(hyp_text, windows)
        elif self.segment_layout == "concat":
            parsed = hyp_to_concat_segment(hyp_text, windows)
        else:
            parsed = hyp_to_target_turn_segments(hyp_text, windows)
        segments = []
        for i, item in enumerate(parsed):
            sentence = self.utt_start_token + item["text"].strip() + self.utt_end_token
            if not sentence.strip():
                continue
            segments.append({
                "utt": f"{cut.id}-{speaker_id}-{i:04d}",
                "cut_id": str(cut.id),
                "speaker": speaker_id,
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": sentence,
                "language": self.language,
                "placement_windows": item["placement_windows"],
            })
        return segments

    def recognize_cutset(self, cutset, text_instance, out_dir=None, save_intermediate=True):
        texts = text_instance
        all_segments = []
        cuts = []
        skipped = 0
        for cut in cutset:
            if cut_anonymize_speaker(cut, default=self.target_speaker_id) is None:
                skipped += 1
                continue
            cuts.append(cut)
        if skipped:
            logger.info(f"Skip {skipped} cuts without anonymize speaker in custom/default")
        if not cuts:
            return texts, all_segments

        short_items, long_items = [], []
        for cut in cuts:
            speakers = cut_speakers(cut)
            if self.decode_anonymize_only:
                target = cut_anonymize_speaker(cut, default=self.target_speaker_id)
                speakers = [target] if target is not None else []
            items = self._items_for_cut(cut, speakers)
            dest = short_items if cut.duration <= 30.0 else long_items
            dest.extend(items)

        decoded_speakers = []

        def _consume(items, batch_size, desc):
            n_batches = (len(items) + batch_size - 1) // batch_size if items else 0
            for b in tqdm(range(n_batches), desc=desc):
                batch = items[b * batch_size:(b + 1) * batch_size]
                try:
                    hyps = self._generate_batch(batch)
                except Exception as err:
                    logger.error(f"DiCoW batch failed ({err}); retrying item-wise")
                    hyps = []
                    for item in batch:
                        try:
                            hyps.append(self._generate_batch([item])[0])
                        except Exception as item_err:
                            logger.error(f"DiCoW failed for cut {item['cut'].id} speaker {item['speaker']}: {item_err}")
                            hyps.append("")
                for item, hyp in zip(batch, hyps):
                    segs = self._segments_from_hyp(item["cut"], hyp, speaker_id=item["speaker"])
                    decoded_speakers.append({
                        "cut_id": str(item["cut"].id),
                        "speaker": item["speaker"],
                        "n_segments": len(segs),
                    })
                    if not segs:
                        logger.warning(f"Empty DiCoW transcript for cut {item['cut'].id} speaker {item['speaker']}")
                        continue
                    for seg in segs:
                        texts.add_instance(
                            sentence=seg["text"],
                            utterance=seg["utt"],
                            speaker=seg["speaker"],
                            language=seg["language"],
                        )
                        all_segments.append(seg)
                    if save_intermediate and out_dir is not None and len(all_segments) % 50 == 0:
                        texts.save_text(out_dir=out_dir)

        logger.info(f"DiCoW ASR on {len(short_items)} short and {len(long_items)} long (cut, speaker) items")
        _consume(short_items, self.batch_size, "DiCoW short")
        _consume(long_items, 1, "DiCoW long")

        if save_intermediate and out_dir is not None:
            texts.save_text(out_dir=out_dir)
            coverage_path = Path(out_dir) / "decoded_speakers.jsonl"
            with coverage_path.open("w", encoding="utf-8") as f:
                for row in decoded_speakers:
                    f.write(json.dumps(row) + "\n")
        return texts, all_segments
