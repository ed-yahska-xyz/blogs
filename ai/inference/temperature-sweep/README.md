# Temperature sweep: one prompt, five temperatures

The same request — the FootyViz match-report prompt from [arg-eng-prompt.md](../arg-eng-prompt.md), a deliberately trap-laden "write only what the digest says" task — sent to `mlx-community/gemma-4-26b-a4b-it-4bit` five times, changing only `temperature`. Full outputs with per-run expectations in [temp-0.0.md](temp-0.0.md), [temp-0.3.md](temp-0.3.md), [temp-0.7.md](temp-0.7.md), [temp-1.2.md](temp-1.2.md), [temp-2.0.md](temp-2.0.md).

## What actually happened

Mistakes are annotated inline in each output file as ~~strikethrough~~ with an *italicized correction*.

| T | Format | Factual accuracy | Character |
|---|--------|------------------|-----------|
| 0.0 | ✅ clean | ✅ **flawless** — all 4 cards with correct minutes, exact names, comeback counted once | Sober, complete, a little listy |
| 0.3 | ✅ clean | ⚠️ one subtle error: folds González's **64'** entrance into the 72' triple substitution — missed on first read, caught only by line-by-line checking | Nearly the same report as T=0.0, reworded at the margins; **identical headline** to T=0.0 |
| 0.7 | ✅ clean | ⚠️ first real errors: moves Gordon's goal from **55' to 51'**; standfirst claims a "**first-half deficit**" (it was 0-0 at HT; the goal came in the 55th) | Livelier; one mangled phrase ("to own the 1-2 lead") |
| 1.2 | ✅ clean | ⚠️ swaps a booking: says Lisandro Martínez was booked **51'** "following the halftime break" (Martínez was 42', first half; 51' was Cristian Romero); repeats the "first-half deficit" error | Confident, fluent — errors are invisible without the digest |
| 2.0 | ❌ collapsed | ❌ n/a | Headline + one standfirst sentence survive, then ~1,300 tokens of multilingual token salad; never emits end-of-turn, dies at `max_tokens` (`finish_reason: length`) |

## Takeaways

1. **Temperature is a factuality dial, not just a "creativity" dial.** On a strict extract-and-write task, T=0.0 was perfect and every step up traded accuracy for prose — even T=0.3 slipped once. The errors were *small and specific* — a sub merged into the wrong minute, a goal moved four minutes, a yellow card reassigned between teammates — exactly the kind of drift temperature causes: the wrong token was always a *plausible* token, and once sampled, the sentence completed around it fluently.

2. **The dangerous zone is the middle, not the top.** T=2.0's failure is self-announcing — no one ships word salad. T=0.7–1.2 produced reports that read perfectly and are wrong in ways only a source-check catches. If the task is factual, the scariest output is the confident one.

3. **Determinism shows at the low end.** T=0.0 and T=0.3 produced the same headline; so did T=0.7 and T=1.2 (a different one). With a sharply-peaked distribution the model keeps re-finding the same high-probability phrasing.

4. **Collapse at T=2.0 is a cascade, not a switch.** The first ~20 tokens were fine — the prompt's conditioning concentrated probability hard enough to survive the flattened distribution — until one low-probability sample landed, became context, and made the next weird token *more* likely. Autoregression amplifies its own mistakes. It also never found the EOS token, burning the full 1,400-token budget: high temperature costs money as well as accuracy.

5. **Temperature is computationally free.** Decode speed was ~35 tok/s at every setting — it's one division over the logits. The forward pass doesn't care; only the sampler changes.

## The case for turning the dial up

Everything above reads as an argument for T=0 — so it's worth being fair to the other direction. The sweep shows real benefits to temperature, visible in these very outputs:

**Better sportswriting.** Compare how the three coherent temperatures handle the same facts. T=0.0 files the bookings like an administrator: *"The referee, Ismail Elfath, issued one yellow card to England's Elliot Anderson in the 37th minute. Argentina received three yellow cards, with..."* — a list wearing a paragraph's clothes. T=0.7 weaves the same card into the match's story: *"The match remained scoreless at half-time following an early yellow card for England's Elliot Anderson."* And T=1.2 finds actual narrative tension for the winner: *"With the match heading toward a draw, Lautaro Martínez scored the decisive late winner"* — where T=0.0 offers the flatter *"The decisive moment arrived in stoppage time."* The T=1.2 report is, sentence for sentence, the best *read* of the five — which is precisely why its two errors needed strikethrough to find.

**Options instead of one answer.** T=0.0 and T=0.3 produced the *identical* headline — regenerate at T=0 and you get the same report back forever. At higher temperatures, each run is a fresh draft: the 0.7/1.2 headline surfaced a detail the low-temperature runs never found (the Atlanta venue). For headlines, angles, and ledes, sampling several candidates and picking the best — best-of-N — only works if N draws differ, and temperature is what makes them differ.

**Escaping the argmax rut.** Greedy decoding can't take the second-best word even when the best one is bland, and it's the setting most prone to degenerate repetition loops. A little temperature is cheap insurance against both — which is why our stress test ran at 0.7: five hundred requests of *identical* T=0 flashcards would have produced near-identical cards from every visit to the same chunk.

**The honest synthesis:** temperature buys variety and voice, and charges for it in verification. The T=1.2 report needed an editor with the digest open; the T=0.0 report needed nothing but was interchangeable with every rerun of itself. Match the dial to the task — extraction and citation at the bottom, drafts and headlines in the middle, and nothing of value at 2.0 — or split the difference the way real pipelines do: sample prose warm, then check facts cold.

*(Method note: one run per temperature — illustrative, not statistical. At T>0 each rerun samples a different report; a rigorous version would run each setting N times and score error rates.)*
