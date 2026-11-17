# Run a single lm-evaluation-harness task against an OpenAI-compatible endpoint.
# Source this from run.sh.
#
# Usage:
#   run_lm_eval <model> <base_url> <task> <num_fewshot> <api> <model_args> <output_dir> [gen_kwargs]
#
# api is `completions` (raw prompt to /v1/completions) or `chat` (the server
# applies the model's chat template, via /v1/chat/completions).
# model_args is the comma-separated key=value string lm_eval expects (without
# `model` or `base_url`; this helper injects those). Pass an empty string if
# there are no extra args.
# gen_kwargs is an optional JSON object merged over the task's generation_kwargs.


run_lm_eval() {
  local model=$1 base_url=$2 task=$3 fewshot=$4 api=$5 model_args=$6 outdir=$7 gen_kwargs=${8:-}
  local lm_model endpoint
  local -a extra=()
  [[ -n "$gen_kwargs" ]] && extra+=(--gen_kwargs "$gen_kwargs")
  case "$api" in
    completions) lm_model=local-completions endpoint=/v1/completions ;;
    chat)
      lm_model=local-chat-completions endpoint=/v1/chat/completions
      extra+=(--apply_chat_template)
      ;;
    *) echo "run_lm_eval: unknown api '${api}'" >&2; return 2 ;;
  esac
  echo "--- :microscope: lm_eval ${task} (${fewshot}-shot, ${api})"
  lm_eval --model "$lm_model" \
    --model_args "model=${model},base_url=${base_url}${endpoint}${model_args:+,$model_args}" \
    --tasks "$task" \
    --num_fewshot "$fewshot" \
    "${extra[@]}" \
    --log_samples \
    --output_path "${outdir}/${task}"
}
