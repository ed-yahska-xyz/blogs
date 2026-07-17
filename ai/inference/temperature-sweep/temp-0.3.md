# Temperature 0.3 — sharpened sampling

**Setup:** the FootyViz match-report prompt from [arg-eng-prompt.md](../arg-eng-prompt.md) (system rules + ENG 1-2 ARG digest), sent to `mlx-community/gemma-4-26b-a4b-it-4bit` with `temperature: 0.3`, `max_tokens: 1400`.

## What to expect

Dividing logits by 0.3 more than triples every gap between candidate tokens before softmax, so the distribution is sharply peaked but no longer a delta function: overwhelmingly the top token, with occasional second choices where the model was genuinely torn. Expect a report structurally almost identical to the T=0.0 run with synonym-level variation — different verbs here and there, maybe a reordered clause. Factual discipline should remain intact; this is the zone most production systems pick for factual generation, trading a sliver of determinism for slightly less robotic prose.

## Model output

> **Fact-check verdict: one subtle error** (annotated below with ~~strikethrough~~ and an *italicized correction*): the report folds Nicolás González's 64th-minute entrance into Argentina's 72nd-minute triple substitution. Everything else — goals, cards, stats, passes, referee — checks out. Note this error was missed on first read and only surfaced on a line-by-line check against the digest.

HEADLINE: Argentina edge England with late winner to reach World Cup final

STANDFIRST: Argentina secured a comeback win to defeat England 2-1 in their World Cup semi-final. The victory extends the winning run for Argentina to five matches.

BODY:
Argentina secured a late winner to defeat England 2-1 at the Mercedes-Benz Stadium, moving closer to glory in their World Cup semi-final. Argentina were behind during the match but fought back to claim the victory.

The contest remained scoreless at half-time following a cautious opening period. England took the lead in the 55th minute when Anthony Gordon scored, assisted by Morgan Rogers. However, Argentina responded in the second half to turn the game around. Enzo Fernández levelled the score in the 85th minute with an assist from Lionel Messi Cuccittini. The decisive blow came in stoppage time as Lautaro Martínez found the net in the 90+2' minute, also assisted by Lionel Messi Cuccittini, to make it 1-2.

The statistical breakdown reflects the decisive nature of the late goals. Argentina finished the match with 64% possession and 15 shots, compared to England's 36% possession and 5 shots. Argentina also recorded an xG of 1.84 against England's 0.53. While England managed 2 shots on target, Argentina registered 5.

The match saw several tactical changes in the final stages. Argentina made multiple substitutions in the 72nd minute, bringing on ~~Nicolás González,~~ Rodrigo De Paul, Gonzalo Montiel, and Nicolás Otamendi. *(González entered in the 64th minute, not as part of the 72nd-minute changes.)* England responded with Ezri Konsa Ngoyo joining the fray at 72' and Daniel Burn and Nico O'Reilly entering in the 82nd minute. In the dying moments, Ivan Toney and Marcus Rashford were introduced for England.

Argentina's victory maintains their form, extending their winning run to five games. Argentina finished with 590 passes and 537 accurate passes, while England recorded 324 passes with 272 being accurate. The referee for the match was Ismail Elfath.

Top performers included Lionel Messi Cuccittini, who provided two assists, and Enzo Fernández, who scored the equaliser. For England, Elliot Anderson received a yellow card in the 37th minute and was rated 7.3.

---
*Run stats: 496 completion tokens · finish_reason stop · 35.2 tok/s*
