# Sourced by run.sh — NCU NVTX filters + launch counts for InternVL3.
# shellcheck disable=SC2034
INTERNVL_NCU_PHASE_ROWS=(
  "Prefill         CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/        1000"
  "Decode          CLOSED_LOOP_INFERENCE/InternVL_LLM_Decode/         2000"
  "Vision_Encoder  CLOSED_LOOP_INFERENCE/InternVL_Vision_Encoder/     1000"
  "MLP_Connector   CLOSED_LOOP_INFERENCE/InternVL_MLP_Connector/      100"
)
