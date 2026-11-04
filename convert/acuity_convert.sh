#!/bin/bash
# Convert the ONNX encoder to an NBG binary for the A733 NPU, INSIDE the Allwinner Acuity container.
#
#   usage (inside the container):  acuity_convert.sh <workdir> <int16|uint8|fp16>
#
# <workdir> must contain:  minilm_npu_128.onnx  and  dataset.txt + calib/emb_NNN.npy,mask_NNN.npy
#                          (see make_calibration.py). Result: <workdir>/wksp/*_nbg_unify/network_binary.nb
#
# Measured on the Orange Pi 4 Pro: int16 = 36 MB, ~81 ms/sequence, cosine ~0.9995 to FP32.
#                                  fp16  = 44 MB, ~2.6 s/sequence (no acceleration).  bf16 does not compile.
set -uo pipefail
W=${1:?workdir}; MODE=${2:-int16}
PEGASUS="python3 /usr/local/acuity_command_line_tools/pegasus.py"
# NPU "v3" (A733). Other Allwinner NPUs use other configs (see the Radxa docs: v1 / v2 / v3).
export VSIMULATOR_SHADER_CORE_COUNT=1 VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
cd "$W" || exit 1

# IMPORTANT: --input-size-list must NOT include the batch dimension. Acuity prepends it itself; with '1,128,384'
# the tensors become 4D/5D and the Transpose layers fail with "Dimension must be 5 but is 4".
$PEGASUS import onnx --model minilm_npu_128.onnx --inputs 'emb add_mask' --input-size-list '128,384#1,1,128' \
  --outputs hidden --output-model minilm.json --output-data minilm.data > import.log 2>&1 || { tail -5 import.log; exit 1; }
$PEGASUS generate inputmeta --model minilm.json --input-meta-output minilm_inputmeta.yml > meta.log 2>&1 || exit 1
# The generated meta treats inputs as images; these are plain tensors.
sed -i 's/category: image/category: undefined/; s/reverse_channel: true/reverse_channel: false/' minilm_inputmeta.yml

EXPORT="--model minilm.json --model-data minilm.data --target-ide-project linux64 --with-input-meta minilm_inputmeta.yml \
  --pack-nbg-unify --optimize $VSIMULATOR_CONFIG --viv-sdk $VIV_SDK --batch-size 1 --output-path wksp/$MODE/$MODE"

if [ "$MODE" = fp16 ]; then
  $PEGASUS export ovxlib $EXPORT --dtype float16 > export.log 2>&1
else
  case $MODE in
    int16) Q=dynamic_fixed_point; T=int16 ;;
    uint8) Q=asymmetric_affine; T=uint8 ;;
    *) echo "mode must be int16, uint8 or fp16"; exit 2 ;;
  esac
  $PEGASUS quantize --model minilm.json --model-data minilm.data --device CPU --with-input-meta minilm_inputmeta.yml \
    --rebuild --model-quantize minilm_$MODE.quantize --quantizer $Q --qtype $T --iterations 64 --batch-size 1 > quant.log 2>&1 \
    || { tail -8 quant.log; exit 1; }
  $PEGASUS export ovxlib $EXPORT --dtype quantized --model-quantize minilm_$MODE.quantize > export.log 2>&1
fi
grep -E "CHECK STATUS|Fatal model" export.log | head -3
find wksp -name network_binary.nb -exec ls -la {} \;
