#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)

home=$PWD
\rm env.sh 2> /dev/null || true
touch env.sh

# VENV install dir
venv_dir=$PWD/venv
export MAMBA_ROOT_PREFIX=".micromamba"  # Local install of micromamba (where the libs/bin will be cached)
mamba_bin="$MAMBA_ROOT_PREFIX/micromamba"

### VERSION

MAMBA_VERSION=1.5.1-0

CUDA_VERSION=12.8
TORCH_VERSION=2.8.0

# Split packages to avoid link script errors blocking everything
# Core packages (essential, no problematic link scripts)
MAMBA_CORE_PACKAGES="sshpass OpenSSH sox libflac tar libacl inotify-tools git-lfs wget curl make cmake ncurses ninja python=3.11 automake libtool boost gxx=12.3.0 gcc=12.3.0 python-sounddevice pkg-config zip"
# Optional packages that may have link script issues (will be installed separately or skipped)
MAMBA_OPTIONAL_PACKAGES="ocl-icd-system nvtop ffmpeg"
# Note: matplotlib removed from conda install to avoid gdk-pixbuf dependency issues
# It will be installed via pip later


cat <<\EOF > env.sh
#!/bin/bash
trigger_new_install() {
  hash_check=".install-hash-$(basename $(dirname $0))"
  stored_hash=$(cat $hash_check 2> /dev/null || >&2 echo "First install of $0")
  current_hash=$(sha256sum "$0" | awk '{print $1}')
  if [ "$current_hash" != "$stored_hash" ] && [ "$NEW_INSTALL_TRIGGER" != "no" ]; then
    [ ! -z $stored_hash ] && echo "$0 has been modified. Triggering new installation..." && echo "Use 'export NEW_INSTALL_TRIGGER=no' do disable this behavoir"
    \rm -rf $@ || true
    echo "$current_hash" > $hash_check
  fi
}
compute_and_write_hash() {
    local line_number=$(grep -n "$FUNCNAME .*$1.*" "$0" | awk -F: '{print $1}')
    sed -i "${line_number}s/  # SHA256:.*//" "$0"
    sed -i "${line_number}s/.*/&  # SHA256: $(sha256sum "$1" | awk '{print $1}')/" "$0"
}
EOF
source ./env.sh

compute_and_write_hash "requirements.txt"  # SHA256: 40809e2b580238712f1c91deb5d46bf31dd6d4c292f6151fe77f92737f978c77
trigger_new_install ".micromamba/micromamba .done-*"

mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making virtual environment =="
  if [ ! -f "$mamba_bin" ]; then
    echo "Downloading micromamba"
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -sS -L "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-linux-64" > "$mamba_bin"
    chmod +x "$mamba_bin"
  fi
  [ -d $venv_dir ] && yes | rm -rf $venv_dir

  echo "Micromamba version:"
  "$mamba_bin" --version

  "$mamba_bin" create -y --prefix "$venv_dir"

  echo "Installing conda dependencies"
  # Remove archive channel if configured to avoid timeout issues
  "$mamba_bin" config --prefix "$venv_dir" remove channels conda-forge/label/archive 2>/dev/null || true
  
  # Step 1: Install core packages (excluding those that pull in gdk-pixbuf)
  echo "Step 1: Installing core packages..."
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge --override-channels $MAMBA_CORE_PACKAGES || exit 1
  
  # Verify Python is installed and working
  if [ ! -f "$venv_dir/bin/python" ]; then
    echo "Error: Python installation failed"
    exit 1
  fi
  "$venv_dir/bin/python" --version || exit 1
  
  # Step 2: Try to install optional packages (may fail due to link script errors, but that's OK)
  echo "Step 2: Installing optional packages (may have link script warnings)..."
  set +e
  # Try to install optional packages, but don't fail if link script errors occur
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge --override-channels $MAMBA_OPTIONAL_PACKAGES 2>&1 | tee /tmp/mamba_optional.log | grep -v "libmamba failed to execute pre/post link script" || true
  optional_status=${PIPESTATUS[0]}
  set -e
  
  # Check if the failure was due to link script errors or something else
  if [ $optional_status -ne 0 ]; then
    if grep -q "libmamba failed to execute pre/post link script" /tmp/mamba_optional.log 2>/dev/null; then
      failed_packages=$(grep "libmamba failed to execute pre/post link script" /tmp/mamba_optional.log | sed 's/.*for //' | sort -u | tr '\n' ' ')
      echo "Warning: Optional packages installation failed due to link script errors for: $failed_packages"
      echo "Core packages are installed. Optional packages can be installed via system package manager if needed."
    else
      echo "Warning: Optional packages installation had issues, but core packages are installed. Continuing..."
    fi
  fi

  touch $mark
fi


if [ -e "$venv_dir" ]; then export PATH="$venv_dir/bin:$PATH"; fi

# Hook Micromamba into the script's subshell (this only lasts for as long as the # script is running)
echo "eval \"\$($mamba_bin shell hook --shell=bash)\"" >> env.sh
echo "micromamba activate $venv_dir" >> env.sh
echo "export LD_LIBRARY_PATH=$venv_dir/lib/:$LD_LIBRARY_PATH" >> env.sh
echo "alias conda=micromamba" >> env.sh
echo "export PIP_REQUIRE_VIRTUALENV=false" >> env.sh
source ./env.sh


mark=.done-cuda
if [ ! -f $mark ]; then
  echo " == Installing cuda =="
  # For CUDA 12.x, use the appropriate channel format
  micromamba install -y --prefix "$venv_dir" -c nvidia -c conda-forge cuda-toolkit==${CUDA_VERSION} || exit 1
  "$venv_dir/bin/nvcc" --version || exit 1
  touch $mark
fi

CUDAROOT=$venv_dir
echo "export CUDAROOT=$CUDAROOT" >> env.sh
source ./env.sh


cuda_version_without_dot=$(echo $CUDA_VERSION | xargs | sed 's/\.//')
mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $TORCH_VERSION for cuda $CUDA_VERSION =="
  # Determine PyTorch CUDA index based on version
  # PyTorch 2.8.0 supports cu126, cu128, cu129 (not cu121)
  # For CUDA 12.8, use cu128 index
  if [[ "$TORCH_VERSION" == "2.8."* ]] && [[ "$CUDA_VERSION" == "12.8" ]]; then
    pytorch_cuda="128"  # PyTorch 2.8.x uses cu128 for CUDA 12.8
  elif [[ "$TORCH_VERSION" == "2.9."* ]] && [[ "$CUDA_VERSION" == "12.8" ]]; then
    pytorch_cuda="128"  # PyTorch 2.9.x uses cu128
  else
    pytorch_cuda="$cuda_version_without_dot"
  fi
  
  version="==$TORCH_VERSION+cu$pytorch_cuda"
  # Match torchvision version to PyTorch version
  # PyTorch 2.8.0 uses torchvision 0.23.0+cu128 (0.19.0 not available in cu128)
  if [[ "$TORCH_VERSION" == "2.8."* ]]; then
    torchvision_version="==0.23.0+cu$pytorch_cuda"
  else
    torchvision_version="==0.24.1+cu$pytorch_cuda"
  fi
  torchaudio_version="==$TORCH_VERSION+cu$pytorch_cuda"
  echo -e "\npip3 install torch$version torchvision$torchvision_version torchaudio$torchaudio_version --force-reinstall --index-url https://download.pytorch.org/whl/cu$pytorch_cuda\n"
  pip3 install torch$version torchvision$torchvision_version torchaudio$torchaudio_version --force-reinstall --index-url https://download.pytorch.org/whl/cu$pytorch_cuda \
    || { echo "Failed to find pytorch $TORCH_VERSION for cuda '$CUDA_VERSION', use specify other torch/cuda version (with variables in install.sh script)"  ; exit 1; }
  python3 -c "import torch; print('Torch version:', torch.__version__)" || exit 1
  echo -e "torch$version\ntorchvision$torchvision_version\ntorchaudio$torchaudio_version" > .pip_constraints.txt
  touch $mark
fi


mark=.done-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip3 install -r requirements.txt -c .pip_constraints.txt  || exit 1
  pip3 install Cython
  touch $mark
fi

echo " == Everything got installed successfully =="
