# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
from argparse import ArgumentParser
import os
parser = ArgumentParser()

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ('true', 'yes', '1')
    return bool(v)

parser.add_argument('--config', default='anon_config.yaml')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--force_compute', default='false', type=str)
parser.add_argument(
    '--dataset', action='append', default=None,
    help='Process only this datasets[].data name (repeatable, or comma-separated). '
         'Default: all datasets in the yaml.',
)
args = parser.parse_args()
args.force_compute = _str_to_bool(args.force_compute)

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
else: # CUDA_VISIBLE_DEVICES more important than the gpu_ids arg
    args.gpu_ids = ",".join([ str(i) for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(","))])

from pathlib import Path
import subprocess
import sys
import torch

# Apply torchaudio compatibility patch before importing speechbrain
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['soundfile', 'sox', 'ffmpeg']
    torchaudio.list_audio_backends = list_audio_backends

from utils import parse_yaml, get_datasets, check_dependencies, setup_logger

logger = setup_logger(__name__)

def shell_run(cmd):
    if subprocess.run(['bash', cmd]).returncode != 0:
        logger.error(f'Failed to bash execute: {cmd}')
        sys.exit(1)


def _in_slurm_array():
    return os.environ.get('SLURM_ARRAY_TASK_ID') is not None


def _maybe_install(script):
    """Skip install.sh in array tasks so parallel jobs do not race the venv."""
    if _in_slurm_array():
        logger.info(f'SLURM array task {os.environ["SLURM_ARRAY_TASK_ID"]}: skip {script}')
        return
    shell_run(script)


def _requested_dataset_names(raw):
    if not raw:
        return None
    names = []
    for item in raw:
        names.extend(part.strip() for part in item.split(',') if part.strip())
    return names


def _filter_datasets(datasets, requested):
    if not requested:
        return datasets
    missing = [name for name in requested if name not in datasets]
    if missing:
        available = ', '.join(datasets) or '(none)'
        logger.error(f'Unknown dataset(s): {", ".join(missing)}. Available: {available}')
        sys.exit(1)
    return {name: datasets[name] for name in requested}


def check_dependencies_in_venv(requirements_file, venv_dir):
    """Run check_dependencies using the given venv's Python (for pipelines with isolated venv)."""
    venv_python = Path(venv_dir) / 'bin' / 'python'
    if not venv_python.exists():
        venv_python = Path(venv_dir) / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        raise FileNotFoundError(f'Python not found in venv: {venv_dir}')
    code = f"import sys; sys.path.insert(0, '.'); from utils import check_dependencies; check_dependencies('{requirements_file}')"
    result = subprocess.run([str(venv_python), '-c', code], cwd=Path(__file__).parent)
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == '__main__':

    config = parse_yaml(Path(args.config))

    # For isolated-venv pipelines: run install first (creates venv), then re-exec with that venv
    if config['pipeline'] in ('sttts', 'sttts_multi', 'sttts_multi_dicow', 'sttts_multi_dicow_sum', 'sttts_multi_dicow_concat', 'sttts_multi_dicow_b3'):
        install_script = 'anonymization/pipelines/sttts/install.sh' if config['pipeline'] == 'sttts' else 'anonymization/pipelines/sttts_multi/install.sh'
        _maybe_install(install_script)
        venv_name = 'venv_sttts' if config['pipeline'] == 'sttts' else 'venv_sttts_multi'
        venv_python = Path(__file__).parent / venv_name / 'bin' / 'python'
        if _in_slurm_array() and not venv_python.exists():
            logger.error(f'{venv_name} missing; run a non-array job first so install.sh is not raced')
            sys.exit(1)
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

    datasets = _filter_datasets(get_datasets(config), _requested_dataset_names(args.dataset))
    logger.info(f'Datasets: {", ".join(datasets) if datasets else "(none)"}')

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    if config['pipeline'] == "mcadams":
        from anonymization.pipelines.mcadams import McAdamsPipeline as pipeline
    elif config['pipeline'] == "sttts":
        _maybe_install('anonymization/pipelines/sttts/install.sh')
        check_dependencies_in_venv('anonymization/pipelines/sttts/requirements.txt',
                                   Path(__file__).parent / 'venv_sttts')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts/download_precomputed_intermediate_repr.sh')
        venv_dir = Path(__file__).parent / 'venv_sttts'
        os.environ['ESPEAK_DATA_PATH'] = str(venv_dir / 'share' / 'espeak-ng-data')
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(venv_dir / 'lib' / 'libespeak-ng.so')
        from anonymization.pipelines.sttts import STTTSPipeline as pipeline
    elif config['pipeline'] == "sttts_multi":
        _maybe_install('anonymization/pipelines/sttts_multi/install.sh')
        check_dependencies_in_venv('anonymization/pipelines/sttts_multi/requirements.txt',
                                   Path(__file__).parent / 'venv_sttts_multi')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts_multi/download_precomputed_intermediate_repr.sh')
        venv_dir = Path(__file__).parent / 'venv_sttts_multi'
        os.environ['ESPEAK_DATA_PATH'] = str(venv_dir / 'share' / 'espeak-ng-data')
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(venv_dir / 'lib' / 'libespeak-ng.so')
        from anonymization.pipelines.sttts_multi import STTTSMultiPipeline as pipeline
    elif config['pipeline'] in ("sttts_multi_dicow", "sttts_multi_dicow_sum", "sttts_multi_dicow_concat", "sttts_multi_dicow_b3"):
        _maybe_install('anonymization/pipelines/sttts_multi/install.sh')
        extra_req = Path(__file__).parent / 'anonymization/pipelines/sttts_multi_dicow/requirements.txt'
        venv_python = Path(__file__).parent / 'venv_sttts_multi' / 'bin' / 'python'
        if extra_req.exists() and not _in_slurm_array():
            pip_cmd = [str(venv_python), '-m', 'pip', 'install', '-r', str(extra_req)]
            if subprocess.run(pip_cmd).returncode != 0:
                logger.error('Failed to install sttts_multi_dicow extra requirements (lhotse, transformers 4.55)')
                sys.exit(1)
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts_multi/download_precomputed_intermediate_repr.sh')
        venv_dir = Path(__file__).parent / 'venv_sttts_multi'
        os.environ['ESPEAK_DATA_PATH'] = str(venv_dir / 'share' / 'espeak-ng-data')
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(venv_dir / 'lib' / 'libespeak-ng.so')
        if config['pipeline'] == "sttts_multi_dicow_sum":
            from anonymization.pipelines.sttts_multi_dicow_sum import STTTSMultiDiCoWSumPipeline as pipeline
        elif config['pipeline'] == "sttts_multi_dicow_concat":
            from anonymization.pipelines.sttts_multi_dicow_concat import STTTSMultiDiCoWConcatPipeline as pipeline
        elif config['pipeline'] == "sttts_multi_dicow_b3":
            from anonymization.pipelines.sttts_multi_dicow_b3 import STTTSMultiDiCoWB3Pipeline as pipeline
        else:
            from anonymization.pipelines.sttts_multi_dicow import STTTSMultiDiCoWPipeline as pipeline
    elif config['pipeline'] == "nac":
        shell_run('anonymization/pipelines/nac/install.sh')
        sys.path.append(os.path.join(os.path.dirname(__file__), 'anonymization/modules/nac/coqui_tts/'))
        if devices[0] == torch.device('cpu'):
            from anonymization.pipelines.nac.nac_pipeline import NACPipeline as pipeline
        else:
            from anonymization.pipelines.nac.nac_pipeline_accelerate import NACPipeline as pipeline
    elif config['pipeline'] == "asrbn":
        shell_run('anonymization/pipelines/asrbn/install.sh')
        check_dependencies('anonymization/pipelines/asrbn/requirements.txt')
        from anonymization.pipelines.asrbn import ASRBNPipeline as pipeline
    elif config['pipeline'] == "multi_asrbn":
        shell_run('anonymization/pipelines/asrbn/install.sh')
        check_dependencies('anonymization/pipelines/asrbn/requirements.txt')
        from anonymization.pipelines.multi_asrbn import MultiASRBNPipeline as pipeline
    elif config['pipeline'] == "ssl":
        shell_run('anonymization/pipelines/ssl/install.sh')
        check_dependencies('anonymization/pipelines/ssl/requirements.txt')
        from anonymization.pipelines.ssl.ssl_pipeline import SSLPipeline as pipeline
    elif config['pipeline'] == "template":
        from anonymization.pipelines.template import TemplatePipeline as pipeline
    else:
        raise ValueError(f"Pipeline {config['pipeline']} not defined/imported")

    logger.info(f'Running pipeline: {config["pipeline"]}')
    p = pipeline(config=config, force_compute=args.force_compute, devices=devices)
    p.run_anonymization_pipeline(datasets)
