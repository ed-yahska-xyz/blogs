#!/usr/bin/env bash
#
# stress-test.sh — soak-test an OpenAI-compatible MLX server without wasting inference.
#
# Instead of throwaway prompts, the work queue is built from the markdown docs in
# this directory. Each request does one of four useful jobs on one chunk of a doc:
#   flashcards — Q/A study cards (presentation material)
#   summary    — 2-sentence TL;DR per section
#   faq        — a likely reader question, answered
#   critique   — find a technical claim that needs a caveat (editorial review)
#
# Every response is logged to results.jsonl with latency, usage, and the server's
# timings block — so the run is also a performance dataset (throughput drift,
# prompt-cache hits, peak memory) you can analyze afterwards.
#
# Usage:
#   ./stress-test.sh                          # 1 hour, sequential, default model
#   DURATION=300 ./stress-test.sh             # 5-minute run
#   CONCURRENCY=4 ./stress-test.sh            # 4 parallel workers (find the knee)
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./stress-test.sh
#
set -uo pipefail

HOST="${HOST:-https://edai.ed-yahska.xyz}"
CERT="${CERT:-secrets/demo-bundle.pem}"
MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-4bit}"
DURATION="${DURATION:-3600}"        # seconds
CONCURRENCY="${CONCURRENCY:-1}"     # parallel workers; 1 = back-to-back soak
MAX_TOKENS="${MAX_TOKENS:-512}"
REQ_TIMEOUT="${REQ_TIMEOUT:-180}"   # per-request curl timeout, seconds

command -v jq >/dev/null || { echo "error: jq is required (brew install jq)"; exit 1; }
[[ -f "$CERT" ]] || { echo "error: client cert bundle '$CERT' not found"; exit 1; }

OUTDIR="stress-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR/chunks"

# ---------------------------------------------------------------- work queue --
# Split every markdown doc here into chunks on '## ' headings.
shopt -s nullglob
docs=(*.md)
[[ ${#docs[@]} -gt 0 ]] || { echo "error: no .md files found to build the work queue"; exit 1; }
for doc in "${docs[@]}"; do
  awk -v dir="$OUTDIR/chunks" -v base="${doc%.md}" '
    /^## /   { n++ }
    { print >> sprintf("%s/%s-%02d.txt", dir, base, n+0) }
  ' "$doc"
done
# portable array fill (macOS ships bash 3.2, which lacks mapfile)
CHUNKS=()
while IFS= read -r f; do CHUNKS+=("$f"); done \
  < <(find "$OUTDIR/chunks" -name '*.txt' -size +300c | sort)
NCHUNKS=${#CHUNKS[@]}
[[ $NCHUNKS -gt 0 ]] || { echo "error: no usable chunks produced"; exit 1; }

TASK_NAMES=(flashcards summary faq critique)
TASK_PROMPTS=(
  "Create 3 question-answer flashcards from this excerpt of a blog post about AI inference. Format each as 'Q:' and 'A:' lines. Excerpt:"
  "Summarize this excerpt of a blog post about AI inference in exactly 2 sentences, suitable for a table of contents. Excerpt:"
  "Write the single most likely question a reader would ask after reading this excerpt of a blog post about AI inference, then answer it in one paragraph. Excerpt:"
  "You are a rigorous technical reviewer. Identify the one technical claim in this excerpt that most deserves a caveat, correction, or citation, and explain why in one paragraph. Excerpt:"
)

echo "server:      $HOST"
echo "model:       $MODEL"
echo "duration:    ${DURATION}s   concurrency: $CONCURRENCY"
echo "work queue:  $NCHUNKS chunks x ${#TASK_NAMES[@]} tasks from: ${docs[*]}"
echo "output:      $OUTDIR/"
echo

END=$(( $(date +%s) + DURATION ))

# ------------------------------------------------------------------- worker --
worker() {
  local wid=$1
  local results="$OUTDIR/results-w${wid}.jsonl"
  local i=$wid ok=0 err=0

  while (( $(date +%s) < END )); do
    local chunk_file="${CHUNKS[$(( i % NCHUNKS ))]}"
    local task_idx=$(( (i / NCHUNKS + i) % ${#TASK_NAMES[@]} ))
    local task="${TASK_NAMES[$task_idx]}"
    local body resp meta http latency ts

    body=$(jq -n --arg model "$MODEL" \
                 --arg prompt "${TASK_PROMPTS[$task_idx]}" \
                 --rawfile chunk "$chunk_file" \
                 --argjson maxtok "$MAX_TOKENS" \
      '{model: $model, max_tokens: $maxtok, temperature: 0.7,
        messages: [{role: "user", content: ($prompt + "\n\n" + $chunk)}]}')

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    resp="$OUTDIR/.resp-w${wid}.json"
    meta=$(curl -sS -m "$REQ_TIMEOUT" --cert "$CERT" -o "$resp" \
                -w '%{http_code} %{time_total}' \
                -H "Content-Type: application/json" \
                -d "$body" \
                "$HOST/v1/chat/completions" 2>/dev/null) || meta="000 0"
    http=${meta%% *}; latency=${meta##* }

    if [[ "$http" == "200" ]] && jq -e '.choices[0].message.content' "$resp" >/dev/null 2>&1; then
      ok=$((ok+1))
      # one JSONL record per request: identity + performance + the useful output
      jq -c --arg ts "$ts" --arg task "$task" --arg chunk "$(basename "$chunk_file")" \
            --argjson latency "$latency" --argjson n "$i" --argjson wid "$wid" \
        '{ts: $ts, n: $n, worker: $wid, task: $task, chunk: $chunk, latency_s: $latency,
          http: 200, usage: .usage, timings: (.timings // null),
          content: .choices[0].message.content}' "$resp" >> "$results"
      # and append the content to a per-task markdown file, tagged with its source
      {
        echo "### [$(basename "$chunk_file" .txt)]"
        jq -r '.choices[0].message.content' "$resp"
        echo
      } >> "$OUTDIR/content-${task}.md"
      printf 'w%d #%-4d %-10s %-34s %6ss  %s tok/s\n' "$wid" "$i" "$task" \
        "$(basename "$chunk_file" .txt)" "$latency" \
        "$(jq -r '.timings.predicted_per_second // "?" | if type=="number" then (.*10 | round)/10 else . end' "$resp")"
    else
      err=$((err+1))
      jq -n -c --arg ts "$ts" --arg task "$task" --argjson n "$i" --argjson wid "$wid" \
            --argjson latency "${latency:-0}" --argjson http "${http:-0}" \
        '{ts: $ts, n: $n, worker: $wid, task: $task, latency_s: $latency, http: $http, error: true}' >> "$results"
      echo "w$wid #$i ERROR http=$http (${err} so far) — backing off 5s"
      sleep 5
    fi
    i=$(( i + CONCURRENCY ))
  done
  echo "worker $wid done: $ok ok, $err errors"
}

# ------------------------------------------------------------------ run + sum --
summarize() {
  echo; echo "== summary =="
  cat "$OUTDIR"/results-w*.jsonl 2>/dev/null | jq -s '
    map(select(.error != true)) as $ok |
    { requests_ok: ($ok | length),
      requests_err: (map(select(.error == true)) | length),
      wall_clock_s: ( (map(.ts) | max) as $z | 0 ),
      total_prompt_tokens: ($ok | map(.usage.prompt_tokens) | add),
      total_completion_tokens: ($ok | map(.usage.completion_tokens) | add),
      cached_prompt_tokens: ($ok | map(.usage.prompt_tokens_details.cached_tokens // 0) | add),
      latency_s: { p50: ($ok | map(.latency_s) | sort | .[length/2|floor] // null),
                   p95: ($ok | map(.latency_s) | sort | .[length*0.95|floor] // null),
                   max: ($ok | map(.latency_s) | max) },
      decode_tok_per_s: { first: ($ok | first.timings.predicted_per_second? // null),
                          last:  ($ok | last.timings.predicted_per_second?  // null),
                          avg:   ($ok | map(.timings.predicted_per_second // empty) | if length>0 then add/length else null end) },
      peak_memory_gb: ($ok | map(.timings.peak_memory // empty) | max) }
    | del(.wall_clock_s)'
  echo "results:  $OUTDIR/results-w*.jsonl"
  echo "content:  $OUTDIR/content-*.md"
}
trap 'echo; echo "interrupted."; summarize; exit 130' INT

for (( w=0; w<CONCURRENCY; w++ )); do worker "$w" & done
wait
summarize
