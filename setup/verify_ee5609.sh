#!/usr/bin/env bash
# EE5609 Project 10 - environment verification (Ubuntu 24.04)
# Checks every tool from setup_ee5609_u24.sh, runs a tiny functional test of
# each, and prints a PASS / FAIL summary with the fix for anything missing.
#
# Usage:  chmod +x verify_ee5609.sh && ./verify_ee5609.sh 2>&1 | tee verify_log.txt
# Nothing is installed or changed; it only checks.

PROJECT_DIR="$HOME/ee5609"
VENV_DIR="$PROJECT_DIR/venv"
LIBDIR=/usr/lib/x86_64-linux-gnu
TMP=$(mktemp -d)
PASS=0; FAIL=0; WARN=0
FIXES=()

ok()   { printf "  [ OK ] %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  [FAIL] %s\n" "$1"; FAIL=$((FAIL+1)); FIXES+=("$1  ->  $2"); }
warn() { printf "  [WARN] %s\n" "$1"; WARN=$((WARN+1)); FIXES+=("(warning) $1  ->  $2"); }

echo "=============================================================="
echo " EE5609 environment check   $(date)"
echo "=============================================================="

echo; echo "--- System ---"
. /etc/os-release 2>/dev/null
echo "  OS      : ${PRETTY_NAME:-unknown}"
echo "  Kernel  : $(uname -r)"
echo "  CPU     : $(nproc) cores,  RAM: $(free -g | awk '/Mem:/{print $2}') GB"
echo "  Disk    : $(df -h "$HOME" | awk 'NR==2{print $4}') free in \$HOME,  $(df -h / | awk 'NR==2{print $4}') free on /"
[ "${VERSION_ID:-}" = "24.04" ] && ok "Ubuntu 24.04" || warn "OS is ${VERSION_ID:-unknown}, scripts assume 24.04" "tell Claude the exact version"
FREE_ROOT=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')
[ "${FREE_ROOT:-0}" -ge 60 ] && ok "Disk space on / >= 60 GB (Vivado needs ~40-60 GB)" \
  || warn "Only ${FREE_ROOT} GB free on / ; Vivado install needs ~40-60 GB" "free space or install Vivado to a larger disk"

echo; echo "--- Build tools / apt packages ---"
for pkg in build-essential git wget curl unzip python3 python3-pip python3-venv \
           iverilog verilator yosys gtkwave locales libncurses6 libtinfo6; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    ok "$pkg $(dpkg -s "$pkg" | awk '/^Version:/{print $2}')"
  else
    bad "$pkg not installed" "sudo apt install -y $pkg"
  fi
done

echo; echo "--- EDA tools (versions + tiny functional test) ---"
cat > "$TMP/mac.v" <<'EOF'
module mac #(parameter W=21) (input clk, input rst, input signed [7:0] a, b,
                               output reg signed [W-1:0] acc);
  always @(posedge clk) if (rst) acc <= 0; else acc <= acc + a*b;
endmodule
EOF
cat > "$TMP/tb.v" <<'EOF'
module tb; reg clk=0, rst=1; reg signed [7:0] a=-128, b=-128; wire signed [20:0] acc;
  mac #(21) u(.clk(clk),.rst(rst),.a(a),.b(b),.acc(acc));
  always #1 clk=~clk;
  initial begin #2 rst=0; #64; if (acc==524288) $display("SIM_OK"); else $display("SIM_BAD %0d",acc); $finish; end
endmodule
EOF
if command -v iverilog >/dev/null; then
  echo "  iverilog : $(iverilog -V 2>&1 | head -n1)"
  if iverilog -o "$TMP/sim" "$TMP/tb.v" "$TMP/mac.v" 2>/dev/null && vvp "$TMP/sim" 2>/dev/null | grep -q SIM_OK; then
    ok "Icarus simulation (32 x (-128*-128) = 524288 in 21 bits)"
  else bad "Icarus installed but test simulation failed" "send verify_log.txt to Claude"; fi
else bad "iverilog missing" "sudo apt install -y iverilog"; fi

if command -v verilator >/dev/null; then
  echo "  verilator: $(verilator --version)"
  verilator --lint-only -Wno-fatal "$TMP/mac.v" >/dev/null 2>&1 && ok "Verilator lint" \
    || bad "Verilator lint failed" "sudo apt install --reinstall -y verilator"
else bad "verilator missing" "sudo apt install -y verilator"; fi

if command -v yosys >/dev/null; then
  echo "  yosys    : $(yosys -V)"
  if yosys -p "read_verilog $TMP/mac.v; synth -top mac; stat" > "$TMP/ys.txt" 2>&1; then
    ok "Yosys synthesis of 21-bit MAC ($(grep -m1 -i 'Number of cells' "$TMP/ys.txt" | awk '{print $NF}') cells)"
  else bad "Yosys synthesis failed" "sudo apt install --reinstall -y yosys"; fi
else bad "yosys missing" "sudo apt install -y yosys"; fi

command -v gtkwave >/dev/null && ok "gtkwave present (GUI, not launched)" || bad "gtkwave missing" "sudo apt install -y gtkwave"

echo; echo "--- Vivado prerequisites ---"
for lib in libtinfo.so.5 libncurses.so.5; do
  if [ -e "$LIBDIR/$lib" ]; then
    tgt=$(readlink -f "$LIBDIR/$lib")
    ok "$lib -> $(basename "$tgt")"
  else
    bad "$lib missing" "re-run step [2/4] of setup_ee5609_u24.sh, or: sudo ln -sf $LIBDIR/${lib%.5}.6 $LIBDIR/$lib"
  fi
done
locale -a 2>/dev/null | grep -qi '^en_US\.utf-\?8$' && ok "en_US.UTF-8 locale" \
  || bad "en_US.UTF-8 locale missing" "sudo locale-gen en_US.UTF-8"

echo; echo "--- Python virtual environment ---"
if [ -x "$VENV_DIR/bin/python" ]; then
  ok "venv at $VENV_DIR ($("$VENV_DIR/bin/python" --version 2>&1))"
  "$VENV_DIR/bin/python" - <<'EOF' > "$TMP/py.txt" 2>&1
import importlib
mods = ["numpy","scipy","matplotlib","pandas","tqdm","torch","torchvision","jupyter_core","notebook"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"PYOK {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"PYBAD {m} {type(e).__name__}: {e}")
try:
    import numpy as np, torch
    from scipy.stats import norm
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = torch.randint(-128,128,(1000,32),dtype=torch.int32)
    s = (x*x).sum(1).max().item()
    assert abs(norm.sf(6) - 9.87e-10) < 1e-11
    plt.plot([1,2]); plt.savefig("/tmp/_ee5609_test.png")
    print(f"PYOK functional numpy/scipy/torch/matplotlib (torch threads={torch.get_num_threads()})")
except Exception as e:
    print(f"PYBAD functional {type(e).__name__}: {e}")
EOF
  while read -r tag name rest; do
    case $tag in
      PYOK)  ok "python: $name $rest" ;;
      PYBAD) if [ "$name" = torch ] || [ "$name" = torchvision ]; then
               bad "python: $name ($rest)" "source $VENV_DIR/bin/activate && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
             elif [ "$name" = functional ]; then
               bad "python functional test ($rest)" "send verify_log.txt to Claude"
             else
               p=$name; [ "$p" = jupyter_core ] || [ "$p" = notebook ] && p=jupyter
               bad "python: $name ($rest)" "source $VENV_DIR/bin/activate && pip install $p"
             fi ;;
    esac
  done < "$TMP/py.txt"
else
  bad "venv not found at $VENV_DIR" "re-run step [3/4] of setup_ee5609_u24.sh"
fi

echo; echo "--- Project folders ---"
for d in montecarlo model rtl synth results report; do
  [ -d "$PROJECT_DIR/$d" ] && ok "$PROJECT_DIR/$d" || bad "folder $PROJECT_DIR/$d missing" "mkdir -p $PROJECT_DIR/$d"
done
[ -f "$PROJECT_DIR/montecarlo/week1_montecarlo.py" ] && ok "week1_montecarlo.py in montecarlo/" \
  || warn "week1_montecarlo.py not in $PROJECT_DIR/montecarlo/" "copy it there to keep the project tidy"

echo; echo "--- Vivado (needed only in Week 4) ---"
VIV=$(command -v vivado || ls -d /tools/Xilinx/Vivado/*/bin/vivado /tools/Xilinx/*/Vivado/bin/vivado 2>/dev/null | head -n1)
if [ -n "$VIV" ]; then
  echo "  found    : $VIV"
  "$VIV" -version 2>/dev/null | head -n2 | sed 's/^/  /'
  command -v vivado >/dev/null && ok "vivado on PATH" \
    || warn "Vivado installed but not on PATH" "add 'source <install>/settings64.sh' to ~/.bashrc, then open a new terminal"
  grep -q settings64.sh "$HOME/.bashrc" && ok "settings64.sh sourced in ~/.bashrc" \
    || warn "settings64.sh not in ~/.bashrc" "echo 'source $(dirname "$(dirname "$VIV")")/settings64.sh' >> ~/.bashrc"
  echo 'puts [llength [get_parts xczu9eg*]]' > "$TMP/p.tcl"
  NP=$(cd "$TMP" && "$VIV" -mode batch -nojournal -nolog -source p.tcl 2>/dev/null | grep -E '^[0-9]+$' | tail -n1)
  [ "${NP:-0}" -gt 0 ] && ok "XCZU9EG device available ($NP part variants)" \
    || bad "XCZU9EG not installed in Vivado" "re-run the installer, Add Design Tools or Devices, tick Zynq UltraScale+ MPSoC"
else
  warn "Vivado not installed yet" "follow the Vivado steps printed at the end of setup_ee5609_u24.sh (only needed for Week 4)"
fi

rm -rf "$TMP" /tmp/_ee5609_test.png
echo
echo "=============================================================="
echo " SUMMARY:  $PASS passed,  $FAIL failed,  $WARN warnings"
echo "=============================================================="
if [ ${#FIXES[@]} -gt 0 ]; then
  echo " To fix:"
  for f in "${FIXES[@]}"; do echo "   - $f"; done
else
  echo " Everything is installed and working."
fi
