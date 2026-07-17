# Temperature 0.7 — the common default

**Setup:** the FootyViz match-report prompt from [arg-eng-prompt.md](../arg-eng-prompt.md) (system rules + ENG 1-2 ARG digest), sent to `mlx-community/gemma-4-26b-a4b-it-4bit` with `temperature: 0.7`, `max_tokens: 1400`.

## What to expect

This is the default of most chat APIs: the distribution keeps most of its shape, so mid-probability tokens get real chances. Expect noticeably livelier sportswriting — varied sentence rhythm, more evocative verbs, a headline with some flair. This is also where the first cracks in factual discipline may appear, and the digest's tripwires tell us where to look: colour that the digest doesn't support ("mounting pressure," "wave after wave"), which the rules explicitly forbid; a shortened player name; or an embellished reading of the one-goal comeback. The report should still be broadly correct — but "broadly" is doing new work at this temperature.

## Model output

> **Fact-check verdict: three errors** (annotated below with ~~strikethrough~~ and *italicized corrections*): a goal moved four minutes earlier, a phantom "first-half deficit", and an invented causal detail about a booking. Note how each wrong token is *plausible* — a nearby minute, a common match narrative — which is exactly how temperature-induced drift looks.

HEADLINE: Argentina edge England with late comeback win in Atlanta

STANDFIRST: Argentina overcome a ~~first-half~~ *second-half* deficit to defeat England in the World Cup semi-finals. *(It was 0-0 at half-time; England's goal came in the 55th minute.)* The victory extends the winning run for the visitors to five games.

BODY:
Argentina secure a place in the World Cup final after fighting back to defeat England 2-1 at the Mercedes-Benz Stadium. After trailing during the match, the visitors secured a comeback win with a decisive late winner to extend their winning run to five games.

The match remained scoreless at half-time following an early yellow card for England's Elliot Anderson in the 37th minute. Argentina's Lisandro Martínez received a yellow card in the 42nd minute ~~to end the first half~~. *(Invented detail — the digest ties no card to the end of the half.)* The deadlock was broken in the ~~51st~~ *55th* minute when Anthony Gordon scored to put England 1-0 ahead, following an assist from Morgan Rogers. *(The 51st minute was Cristian Romero's yellow card, whose booking this report omits.)*

Argentina responded to the deficit, and in the 85th minute, Enzo Fernández scored to make it 1-1 after receiving an assist from Lionel Messi Cuccittini. The score remained level until the 90+2' minute, when Lautaro Martínez found the net through an assist from Lionel Messi Cuccittini to own the 1-2 lead. 

Argentina controlled much of the play, finishing with 64% possession and 15 shots compared to England's 5. The visitors also recorded 5 shots on target and 6 corners, while England managed 2 shots on target and 1 corner. Argentina's xG stood at 1.84 to England's 0.53.

In the final minutes, Rodrigo De Paul received a yellow card in the 90+4' minute. England made late substitutions with Ivan Toney and Marcus Rashford entering the pitch.

The final score reflected a clinical performance from the visitors, who completed the comeback to move one step closer to the title.

England lineup: Jordan Pickford, Diop Djed-Hotep Spence, Addji Guéhi, John Stones, Reece James, Elliot Anderson, Declan Rice, Anthony Gordon, Jude Bellingham, Morgan Rogers, Harry Kane.

Argentina lineup: Damián Martínez Romero, Nicolás Tagliafico, Lisandro Martínez, Cristian Romero, Nahuel Molina Lucero, Leandro Paredes, Julián Álvarez, Alexis Mac Allister, Enzo Fernández, Giuliano Simeone Baldini, Lionel Messi Cuccittini.

---
*Run stats: 510 completion tokens · finish_reason stop · 35.2 tok/s*
