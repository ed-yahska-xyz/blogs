# Temperature 1.2 — above the model's native distribution

**Setup:** the FootyViz match-report prompt from [arg-eng-prompt.md](../arg-eng-prompt.md) (system rules + ENG 1-2 ARG digest), sent to `mlx-community/gemma-4-26b-a4b-it-4bit` with `temperature: 1.2`, `max_tokens: 1400`.

## What to expect

Above T=1.0 we're flattening the distribution beyond what the model learned: tokens the model considers clearly worse now get meaningful probability mass. Expect creative-register prose — metaphors, dramatic framing — and a real chance of rule violations: invented context or momentum, "Messi" instead of "Lionel Messi Cuccittini," a "fought back twice" miscount, the missing player appearing, or drift from the required HEADLINE/STANDFIRST/BODY format. Each generated token also conditions everything after it, so one odd word choice can pull the rest of the report somewhere strange. If the T=0.7 run had hairline cracks, this is where they should widen visibly.

## Model output

> **Fact-check verdict: two errors** (annotated below with ~~strikethrough~~ and *italicized corrections*): the same phantom "first-half deficit" as the T=0.7 run, and a booking reassigned from one Argentina defender to another with the wrong half attached. The rest — goals, stats, all ten substitutes, both lineups, top performers — is accurate, which is what makes the two errors dangerous: nothing about the prose signals them.

HEADLINE: Argentina edge England with late comeback win in Atlanta

STANDFIRST: Argentina overcome a ~~first-half~~ *second-half* deficit to defeat England 2-1 at the Mercedes-Benz Stadium. *(It was 0-0 at half-time; the deficit began with England's 55th-minute goal.)* The victory extends the winning run for the visitors to five matches.

BODY:
Argentina secure a place in the World Cup final after a 2-1 comeback win against England at the Mercedes-Benz Stadium. The visitors were behind during the match but fought back to claim the victory in stoppage time.

After a scoreless first half, Anthony Gordon put England ahead in the 55th minute following an assist from Morgan Rogers. The England lead lasted until the 85th minute when Enzo Fernández equalised for Argentina, assisted by Lionel Messi Cuccittini. With the match heading toward a draw, Lautaro Martínez scored the decisive late winner in the 90+2nd minute, again assisted by Lionel Messi Cuccittini.

Argentina controlled much of the play with 64% possession compared to 36% for England. The statistical gap was evident as Argentina recorded 15 shots with 5 on target, while England managed 5 shots with 2 on target. Argentina also finished with an xG of 1.84 to England's 0.53.

The match saw several disciplinary actions and tactical changes. Elliot Anderson received a yellow card for tripping in the 37th minute. Following the halftime break, ~~Lisandro Martínez~~ *Cristian Romero* was booked in the 51st minute for holding. *(Lisandro Martínez's booking, also for holding, came in the 42nd minute — before half-time. The model merged two similar bookings into one wrong sentence.)* 

Substitutions heavily influenced the final period. Argentina introduced Nicolás González, Rodrigo De Paul, Gonzalo Montiel, Nicolás Otamendi, and Lautaro Martínez to fresh legs. England responded with moves for Ezri Konsa Ngoyo, Daniel Burn, Nico O'Reilly, Ivan Toney, and Marcus Rashford. Rodrigo De Paul received a yellow card for unsportsmanlike conduct in the 90+4th minute.

The scoring sequence saw England lead 1-0 before Argentina fought back to level at 1-1 and then take a 1-2 lead. Top performers for the winning side included Lionel Messi Cuccittini, who provided two assists, and Enzo Fernández. For England, Elliot Anderson and Declan Rice earned high ratings.

England's lineup featured Jordan Pickford, Diop Djed-Hotep Spence, Addji Guéhi, John Stones, Reece James, Elliot Anderson, Declan Rice, Anthony Gordon, Jude Bellingham, Morgan Rogers, and Harry Kane. Argentina's lineup consisted of Damián Martínez Romero, Nicolás Tagliafico, Lisandro Martínez, Cristian Romero, Nahuel Molina Lucero, Leandro Paredes, Julián Álvarez, Alexis Mac Allister, Enzo Fernández, Giuliano Simeone Baldini, and Lionel Messi Cuccittini.

---
*Run stats: 577 completion tokens · finish_reason stop · 35.2 tok/s*
