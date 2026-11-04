from pathlib import Path

import torch

from .anonymization.random_anon import RandomAnonymizer
from .anonymization.pool_anon import PoolAnonymizer
from .anonymization.gan_anon import GANAnonymizer
from .speaker_embeddings import SpeakerEmbeddings
from utils import setup_logger

logger = setup_logger(__name__)


def _slice_embeddings(emb: SpeakerEmbeddings, identifiers: list) -> SpeakerEmbeddings:
    ids = list(identifiers)
    vecs = [emb.get_embedding_for_identifier(i).reshape(-1) for i in ids]
    spks = [emb.original_speakers[int(emb.identifiers2idx[i])] for i in ids]
    genders = [emb.genders[int(emb.identifiers2idx[i])] for i in ids]
    out = SpeakerEmbeddings(vec_type=emb.vec_type, emb_level=emb.emb_level, device=emb.device)
    out.set_vectors(ids, torch.stack(vecs, dim=0), spks, genders)
    return out

class SpeakerAnonymization:

    def __init__(self, vectors_dir, device, settings, results_dir=None, save_intermediate=True, force_compute=False):
        self.vectors_dir = vectors_dir
        self.device = device
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_anonymization', False)

        self.vec_type = settings['vec_type']
        self.emb_level = settings['emb_level']

        if results_dir:
            self.results_dir = results_dir
        elif 'anon_results_path' in settings:
            self.results_dir = settings['anon_results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        self.anonymizer = self._load_anonymizer(settings)

    @property
    def suffix(self):
        return self.anonymizer.suffix

    def anonymize_embeddings(self, speaker_embeddings, dataset_name):
        """Reuse cached GAN voices by identifier; draw only for missing ids.

        concat / sum / b3 share ``anon_results_path`` and the same mix ids, so
        the same original ECAPA keeps the same GAN vector across pipelines.
        ``speaker_embeddings.new`` is ignored: a freshly built mix-level
        wrapper must not redraw voices that are already on disk.
        """
        dataset_results_dir = Path(self.results_dir) / dataset_name if self.save_intermediate else None
        requested = speaker_embeddings.get_utt_list()

        cached = None
        cache_ok = (
            dataset_results_dir is not None
            and dataset_results_dir.exists()
            and any(dataset_results_dir.iterdir())
            and (dataset_results_dir / "speaker_vectors.pt").exists()
        )
        if cache_ok and not self.force_compute:
            cached = SpeakerEmbeddings(
                vec_type=self.vec_type, emb_level=self.emb_level, device=self.device
            )
            cached.load_vectors(dataset_results_dir)

        if cached is not None:
            missing = [i for i in requested if i not in cached.identifiers2idx]
            if not missing:
                logger.info(
                    f"Reuse {len(requested)} cached GAN voices from {dataset_results_dir}"
                )
                return cached
            logger.info(
                f"Reuse {len(requested) - len(missing)} cached GAN voices; "
                f"draw {len(missing)} new from {dataset_results_dir}"
            )
            drawn = self.anonymizer.anonymize_embeddings(
                _slice_embeddings(speaker_embeddings, missing), emb_level=self.emb_level
            )
            cached.add_vectors(
                drawn.get_utt_list(), drawn.vectors, drawn.original_speakers, drawn.genders
            )
            if self.save_intermediate:
                cached.save_vectors(dataset_results_dir)
            return cached

        logger.info("Anonymize speaker embeddings...")
        anon_embeddings = self.anonymizer.anonymize_embeddings(
            speaker_embeddings, emb_level=self.emb_level
        )
        if self.save_intermediate:
            anon_embeddings.save_vectors(dataset_results_dir)
        return anon_embeddings

    def _load_anonymizer(self, settings):
        anon_settings = settings['anon_settings']
        anon_method = anon_settings.get('anon_method', 'gan')
        anon_settings['save_intermediate'] = self.save_intermediate or anon_settings['save_intermediate']

        if anon_method == 'random':
            model = RandomAnonymizer(device=self.device, **anon_settings)

        elif anon_method == 'pool':
            model = PoolAnonymizer(device=self.device, **anon_settings)

        elif anon_method == 'gan':
            model = GANAnonymizer(device=self.device, **anon_settings)
        else:
            raise ValueError(f'Unknown anonymization method {anon_method}')

        logger.info(f'Model type of anonymizer: {type(model).__name__}')
        return model
