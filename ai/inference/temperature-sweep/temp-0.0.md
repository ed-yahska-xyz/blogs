# Temperature 0.0 — greedy decoding

**Setup:** the FootyViz match-report prompt from [arg-eng-prompt.md](../arg-eng-prompt.md) (system rules + ENG 1-2 ARG digest), sent to `mlx-community/gemma-4-26b-a4b-it-4bit` with `temperature: 0.0`, `max_tokens: 1400`.

## What to expect

At T=0 the softmax collapses to argmax: the sampler takes the single most probable token at every step, so the run is deterministic (send it again, get essentially the same report back). This should be the most disciplined run of the sweep — best adherence to the digest-only rules, exact names copied verbatim, the comeback counted once, no invented momentum. The trade-off is prose: greedy text tends to be safe, formulaic, and occasionally repetitive, because the model can never take the second-best word even when the best one is bland. The known failure mode to watch for is degenerate repetition — greedy decoding is where models get stuck in loops.

## Model output

> **Fact-check verdict: no factual errors.** Every minute, name, card, and stat matches the digest — all four yellows correctly attributed, exact long names copied, the comeback counted once. The only blemish is *"a cautious opening period"*, an inference the digest doesn't state (the rules forbid inferred momentum), though it invents no fact.

HEADLINE: Argentina edge England with late winner to reach World Cup final

STANDFIRST: Argentina secured a comeback win to defeat England 2-1 in their World Cup semi-final. The victory extends the winning run for Argentina to five games.

BODY:
Argentina secured a late winner to defeat England 2-1 at the Mercedes-Benz Stadium, moving one step closer to glory. After trailing during the match, Argentina fought back to claim the victory in the dying moments of the game.

The contest remained scoreless at half-time following a cautious opening period. England took the lead in the 55th minute when Anthony Gordon scored, assisted by Morgan Rogers, to make it 1-0. However, Argentina responded in the 85th minute as Enzo Fernández found the net following an assist from Lionel Messi Cuccittini to level the score.

The decisive moment arrived in stoppage time. In the 90+2' minute, Lautaro Martínez scored to make it 1-2, ensuring Argentina maintained their winning run to five games.

Argentina dominated much of the statistical battle, recording 64% possession compared to England's 36%. The visitors registered 15 shots with 5 on target, while England managed 5 shots with 2 on target. The xG count stood at 1.84 for Argentina and 0.53 for England. Argentina also completed 537 accurate passes from 590 attempts, while England completed 272 accurate passes from 324 attempts.

The referee, Ismail Elfath, issued one yellow card to England's Elliot Anderson in the 37th minute. Argentina received three yellow cards, with Lisandro Martínez cautioned in the 42nd minute, Cristian Romero in the 51st minute, and Rodrigo De Paul in the 90+4' minute.

Argentina's lineup featured Damián Martínez Romero, Nicolás Tagliafico, Lisandro Martínez, Cristian Romero, Nahuel Molina Lucero, Leandro Paredes, Julián Álvarez, Alexis Mac Allister, Enzo Fernández, Giuliano Simeone Baldini, and Lionel Messi Cuccittini. England's lineup featured Jordan Pickford, Diop Djed-Hotep Spence, Addji Guéhi, John Stones, Reece James, Elliot Anderson, Declan Rice, Anthony Gordon, Jude Bellingham, Morgan Rogers, and Harry Kane.

Top performers included Lionel Messi Cuccittini with an 8.0 rating and two assists, and Enzo Fernández with a 7.6 rating and one goal. For England, Elliot Anderson earned a 7.3 rating.

---
*Run stats: 549 completion tokens · finish_reason stop · 35.9 tok/s*
