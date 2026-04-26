# Sourced by run.sh — NCU NVTX filters + launch counts for LLaVA.
# shellcheck disable=SC2034
LLAVA_MODEL_PREFIX="${LLAVA_MODEL_PREFIX:-LLaVA}"
P="$LLAVA_MODEL_PREFIX"
LLAVA_NCU_PHASE_ROWS=(
  "Prefill         CLOSED_LOOP_INFERENCE/${P}_LLM_Prefill/        200"
  "Decode          CLOSED_LOOP_INFERENCE/${P}_LLM_Decode/         200"
  "Vision_Encoder  CLOSED_LOOP_INFERENCE/${P}_Vision_Encoder/     200"
  "MLP_Connector   CLOSED_LOOP_INFERENCE/${P}_MLP_Connector/      50"
)
