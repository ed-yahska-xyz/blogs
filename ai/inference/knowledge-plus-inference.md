---
marp: true
theme: default
paginate: true
title: "Knowledge + Inference is Power"
description: "Why access to inference is the defining equity question of the AI era"
---

# Knowledge + Inference is Power

## Why access to AI may not follow the internet's open path

Akshay Shinde

<!--
Framing for the talk: this is a story in three acts.
Act 1: the internet era and why "knowledge is power" worked.
Act 2: why we got lucky — the internet's openness was a choice, not an inevitability.
Act 3: AI is at the same fork today, and the defaults are pointing the other way.
-->

---

# The old equation (early 2000s)

## **Knowledge is Power**

- The internet's promise: put the world's knowledge within reach of anyone with a connection
- Google's founding mission: *"organize the world's information and make it universally accessible and useful"*
- The fight of that era was the **digital divide** — getting people *connected*
- Once you were online, the knowledge itself was essentially **free**

<!--
Key point: in the internet era, the bottleneck was the pipe, not the content.
Search was free, Wikipedia was free, documentation was free.
The equity fight was about connectivity, not about the price of a query.
-->

---

# The divide we fought then: access to the pipe

- Pew Research's first reports (2000): lower-income households, minorities, and seniors were far less likely to be online¹
- Income remained one of the strongest predictors of internet use throughout the 2000s²
- But the divide was **binary and closable**: once connected, a student in a public library had the same Google, the same Wikipedia, as a hedge fund analyst

> The playing field leveled *at the point of access*.

<!--
¹ Pew "Digital differences" (2012), summarizing trends since 2000
² Pew found age, education, and income the strongest predictors of internet use
The crucial property: the internet did not meter knowledge per-question.
-->

---

# We got lucky: the internet was born open

Three choices that were **not inevitable**:

| Year | Choice | Consequence |
|------|--------|-------------|
| 1993 | CERN puts the Web in the **public domain**, royalty-free — *"CERN relinquishes all intellectual property rights to this code"* | No one could own or tax the Web itself |
| 1995 | NSFNET backbone decommissioned; NSF pushes **competing commercial backbones** with open interconnection points | No single gatekeeper to the network |
| 2000s | Search & knowledge platforms choose **ad-supported, free-at-point-of-use** models | Knowledge access decoupled from ability to pay |

<!--
CERN's April 30, 1993 statement is the hinge of this whole talk.
Berners-Lee proposed the Web in 1989; CERN could have licensed it.
By late 1993 there were 500+ web servers. Mosaic then made it explode.
NSFNET: academic backbone handed off to multiple competing commercial carriers with open network access points — not one company.
-->

---

# The counterfactual

## Imagine Google charged **$100/month** to search the internet.

- Homework happens only in households that can afford it
- The self-taught programmer, the immigrant learning a language, the patient researching a diagnosis — **priced out**
- Wikipedia, Stack Overflow, open-source — most of it never forms, because the audience that built it couldn't afford to browse

**That world sounds absurd — only because we never lived in it.**

<!--
This is the emotional core of the deck. Sit on this slide.
The point: free knowledge access feels like a law of nature, but it was a
business-model choice layered on an open-technology choice.
-->

---

# The new equation

## **Knowledge + Inference is Power**

- Knowledge is now abundant — the internet solved that
- What's scarce is **inference**: the compute-hungry act of *applying* a model to your problem
- Raw knowledge no longer differentiates; the ability to **reason over it at scale** does
- Every AI answer, every agent step, every generated line of code is a **metered act of inference**

> The bottleneck moved from the pipe to the thinking.

<!--
Define inference for a general audience: training happens once; inference is
every single use of the model. It has a real marginal cost (GPUs, energy),
unlike serving a cached web page. That marginal cost is why this era's
economics default to metering in a way the web's never did.
-->

---

# This time, the defaults point the other way

- Frontier models launched **closed-weight**, behind APIs and subscriptions
- Access is already tiered: free → $20/month → **$200/month** pro tiers, with the true frontier increasingly priced above that³
- A $200/month subscription used heavily can consume **thousands of dollars** of underlying compute — labs subsidize it, and subsidies end⁴
- Training and serving frontier models requires capital measured in **billions**, concentrating capability in a handful of labs and clouds

**The internet's marginal cost rounded to zero. Inference doesn't.**

<!--
³ ChatGPT tier ladder: free / $20 Plus / $200 Pro (OpenAI Help Center)
⁴ TechSpot analysis: a fully-utilized $200 Pro plan ≈ $14,000 at API rates —
today's flat-rate access to the frontier is a loss-leader, not a stable state.
Agentic workloads make it worse: agent runs can consume ~1000x the tokens
of a single prompt.
-->

---

# The inference divide is already measurable

- IMF: **~40% of global employment** is exposed to AI — **60% in advanced economies vs 26% in low-income countries**⁵
- IMF's conclusion: in most scenarios, AI **worsens inequality** — between workers who can harness it and those who can't, and between nations with compute infrastructure and those without
- Within countries: the $200/month tier question is the $100 Google question, **already here**

> Who gets to think with machines — and who thinks alone?

<!--
⁵ IMF analysis, Jan 2024 (Georgieva): "AI Will Transform the Global Economy.
Let's Make Sure It Benefits Humanity."
Note the inversion vs the internet: low-income countries are LESS exposed
only because they're excluded from the upside too — infrastructure gap.
-->

---

# The counter-currents (reasons for hope)

- **Inference is getting radically cheaper**: GPT-3.5-level inference fell **~280×** in 2 years — $20 → $0.07 per million tokens (Stanford AI Index 2025)⁶
- Hardware cost **−30%/year**, energy efficiency **+40%/year**⁶
- **Open-weight models** (Llama, DeepSeek, Mistral, Qwen) closed the benchmark gap with closed models from **8% → 1.7%** in one year⁶
- DeepSeek R1 showed frontier-class reasoning can be **replicated and released openly**
- Sam Altman: cost per unit of intelligence has dropped **~10× per year**; "intelligence too cheap to meter is well within grasp"⁷

<!--
⁶ Stanford HAI AI Index 2025
⁷ Altman, "The Gentle Singularity" — but note the tension: the same labs
promising "too cheap to meter" also say intelligence will be "a utility...
people buy it from us on a meter." Cheap ≠ open. Electricity is cheap and
metered — and people still get disconnected.
Yesterday's frontier gets cheap fast. But power accrues to whoever holds
TODAY'S frontier. The divide may be a moving frontier, not a closing gap.
-->

---

# Two futures

| | **Metered intelligence** | **Open intelligence** |
|---|---|---|
| Model weights | Closed, API-only | Open-weight, self-hostable |
| Best capability goes to | Highest payers first | Everyone, with a lag |
| Analogy | Cable TV, Bloomberg terminal | The Web, Wikipedia, Linux |
| Divide | Compounds (frontier stays paywalled) | Narrows (frontier diffuses) |

The internet got its CERN moment in 1993.
**AI's CERN moment is still an open question.**

<!--
Honest caveat to raise verbally: full openness has real safety trade-offs
that publishing the Web's code didn't. The point isn't "open everything" —
it's that broad access to inference must be a design goal, not an accident.
-->

---

# What would "born open" look like for AI?

- **Open weights** for capable (even if not frontier) models — the Linux tier of intelligence
- **Public/academic compute** — an NSFNET for inference (national research clouds, subsidized capacity)
- **Free tiers treated as infrastructure**, not marketing — the way search stayed free
- **Falling costs passed to users**, not captured entirely as margin
- Policy attention on **inference access**, not just model safety

> We didn't plan the open internet's equity effects — we inherited them.
> With AI, we have to **choose** them.

---

# The takeaway

## Knowledge is power → **Knowledge + Inference is power**

1. The internet made knowledge free — by *choices*, not by nature
2. AI makes inference the new scarce input to power
3. The default trajectory is metered, tiered, and concentrated
4. Cost collapse and open weights make the open path *possible* — not guaranteed

**The question of our decade: will inference be a utility everyone can afford — or a subscription some can't?**

---

# Sources

- CERN — [30 years of a free and open Web](https://home.cern/news/news/computing/30-years-free-and-open-web) · [CERN puts the Web in the public domain (1993)](https://timeline.web.cern.ch/cern-puts-world-wide-web-public-domain)
- NSF — [Birth of the Commercial Internet](https://www.nsf.gov/impacts/internet) · [NSFNET history (IBM)](https://www.ibm.com/history/nsfnet)
- Pew Research Center — [Digital differences (2012)](https://www.pewresearch.org/internet/2012/04/13/digital-differences/) · [Digital divide persists (2021)](https://www.pewresearch.org/short-reads/2021/06/22/digital-divide-persists-even-as-americans-with-lower-incomes-make-gains-in-tech-adoption/)
- IMF — [AI Will Transform the Global Economy. Let's Make Sure It Benefits Humanity (2024)](https://www.imf.org/en/blogs/articles/2024/01/14/ai-will-transform-the-global-economy-lets-make-sure-it-benefits-humanity) · [CNBC coverage](https://www.cnbc.com/2024/01/15/imf-warns-ai-to-hit-almost-40percent-of-global-employment-worsen-inequality.html)
- Stanford HAI — [The 2025 AI Index Report](https://hai.stanford.edu/ai-index/2025-ai-index-report)
- OpenAI — [ChatGPT Pro tiers](https://help.openai.com/en/articles/9793128-what-is-chatgpt-pro) · TechSpot — [$200 subscription vs ~$14,000 compute](https://www.techspot.com/news/112759-openai-anthropic-cant-afford-have-everyone-use-ai.html)
- Sam Altman — [The Gentle Singularity](https://blog.samaltman.com/the-gentle-singularity) · Fortune — ["Intelligence too cheap to meter"](https://fortune.com/2025/07/23/sam-altman-artificial-intelligence-too-cheap-jobs/)
- Berkeley CMR — [How Open-Source AI Will Challenge Closed-Model Giants (2026)](https://cmr.berkeley.edu/2026/01/the-coming-disruption-how-open-source-ai-will-challenge-closed-model-giants/)
