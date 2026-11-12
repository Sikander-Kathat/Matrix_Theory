#!/usr/bin/env bash
# EE5609 Project 10 - Statistical Accumulator Sizing
# Clean environment setup for Ubuntu 24.04 (noble)
#
#   Step 0: clean up leftovers from the earlier 22.04 attempt
#   Step 1: system packages (Icarus Verilog, Verilator, Yosys, GTKWave, build tools)
#   Step 2: Vivado compatibility libraries (libtinfo5 / libncurses5 for 24.04)
#   Step 3: Python virtual environment (NumPy, SciPy, Matplotlib, PyTorch CPU)
#   Step 4: project folders + verification
#
# Usage:  chmod +x setup_ee5609_u24.sh && ./setup_ee5609_u24.sh
# Safe to re-run.

set -e

PROJECT_DIR="$HOME/ee5609"
VENV_DIR="$PROJECT_DIR/venv"
LIBDIR=/usr/lib/x86_64-linux-gnu
NCURSES_POOL=http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/

echo "=== [0/4] Cleaning up leftovers from the earlier attempt ==="
# Failed .deb downloads in /tmp
rm -f /tmp/libtinfo5_*.deb /tmp/libncurses5_*.deb
# Old 22.04 setup script (replaced by this one)
rm -f "$HOME/setup_ee5609.sh"
# Half-built Python env, if the old script got that far (recreated below)
rm -rf "$VENV_DIR"
echo "Cleanup done."

echo "=== [1/4] System packages ==="
sudo apt update
sudo apt install -y \
    build-essential git wget curl unzip \
    python3 python3-pip python3-venv \
    iverilog verilator yosys gtkwave \
    libncurses6 libtinfo6 locales
sudo locale-gen en_US.UTF-8

echo "=== [2/4] Vivado compatibility libraries (libtinfo5 / libncurses5) ==="
if [ -e "$LIBDIR/libtinfo.so.5" ] && [ -e "$LIBDIR/libncurses.so.5" ]; then
    echo "Already present - skipping."
else
    # Try the real Ubuntu 22.04 packages first (latest 6.3 build in the archive)
    T=$(curl -s "$NCURSES_POOL" | grep -o 'libtinfo5_6\.3-[^"]*_amd64\.deb'   | sort -uV | tail -1 || true)
    N=$(curl -s "$NCURSES_POOL" | grep -o 'libncurses5_6\.3-[^"]*_amd64\.deb' | sort -uV | tail -1 || true)
    if [ -n "$T" ] && [ -n "$N" ]; then
        echo "Installing $T and $N"
        cd /tmp
        wget -q "$NCURSES_POOL$T" "$NCURSES_POOL$N"
        sudo apt install -y "./$T" "./$N"
        rm -f "/tmp/$T" "/tmp/$N"
        cd - >/dev/null
    else
        # Fallback: point the v5 names at the system's v6 libraries
        echo "22.04 packages not found in archive - using compatibility symlinks."
        sudo ln -sf "$LIBDIR/libtinfo.so.6"   "$LIBDIR/libtinfo.so.5"
        sudo ln -sf "$LIBDIR/libncurses.so.6" "$LIBDIR/libncurses.so.5"
    fi
fi

echo "=== [3/4] Python virtual environment ==="
mkdir -p "$PROJECT_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install numpy scipy matplotlib pandas tqdm jupyter
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== [4/4] Project folders + verification ==="
mkdir -p "$PROJECT_DIR"/{montecarlo,model,rtl,synth,results,report}

python -c "import numpy, scipy, torch, torchvision, matplotlib; print('Python stack OK - torch', torch.__version__)"
iverilog -V 2>&1 | head -n 1
verilator --version
yosys -V
ls -l "$LIBDIR/libtinfo.so.5" "$LIBDIR/libncurses.so.5"

echo
echo "All open-source tools installed in $PROJECT_DIR"
echo "Activate the Python env any time with:  source $VENV_DIR/bin/activate"
echo
cat <<'EOF'
=====================================================================
 NEXT: install Vivado ML Standard (free) manually
=====================================================================
 1. Log in to your AMD account and open:
      https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools.html
    Pick the newest release whose supported-OS list includes Ubuntu 24.04.
 2. Download the "AMD Unified Installer for Linux" (.bin web installer).
 3. Run it:
      chmod +x FPGAs_AdaptiveSoCs_Unified_*_Lin64.bin
      sudo ./FPGAs_AdaptiveSoCs_Unified_*_Lin64.bin
 4. In the installer choose:
      Product  -> Vivado
      Edition  -> Vivado ML Standard
      Devices  -> untick everything EXCEPT "Zynq UltraScale+ MPSoC"
      Location -> /tools/Xilinx (default)
 5. Add to ~/.bashrc (adjust the version folder):
      source /tools/Xilinx/Vivado/<version>/settings64.sh
 6. Check:  vivado -version
=====================================================================
EOF
