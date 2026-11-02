# Codebook U1: fear on the user's own scale (25 Oct 2026)

Derived from the user's hand ratings of 100 headlines (`my-ratings/MY_100_final.csv`, 21 corrections agreed after a context check). Use it for any future labels meant to match the user. It is calmer than codebook G1: about half a point lower on average.

**Question:** reading only this headline, how threatened or anxious would the user feel?

| fear | the user's rule | examples (user's final score) |
|---|---|---|
| 0 | no threat to people; sport, procedure, arts; **rescued / found safe / escaped**; figurative harm words; animals only; history pieces | "bushwalkers missing … found safe" 0 · "passengers escape plane inferno" 0 · "brown fights for life after euro poll thrashing" 0 · "australias last african elephant dies" 0 |
| 1 | mild: an investigation or court case after harm; aftermath or costs; a contained fire; drought; tension or threats far away; crime with a victim but no ongoing danger | "thai probe hits hurdle…" 1 · "firefighters monitor teelah blaze" 1 · "man who killed two women … jailed" 1 · "murdered keem's ex husband en route to qld" 1 |
| 2 | clear: **a single accidental death**; a missing person, search ongoing; a bombing with likely casualties; injuries; a child's suspicious death; hundreds of local jobs lost; an ongoing local emergency | "australian man falls to his death on nz hunting trip" 2 · "river search fails to find missing driver" 2 · "holden to axe hundreds of staff in adelaide" 2 · "queensland bushfires emergency destroys homes" 2 |
| 3 | alarm: **several deaths, or violent deaths** (bombings, massacres); many people missing with live danger; mass-casualty disasters | "police fear for the safety of 11 missing tourists" 3 · "massacre at movie theatre" 3 · "car bomb kills 5 in iraqi market" 3 · "yangtze ferry disaster families anxious…" 3 |

**Tie-breaks**
- Use context the reader would have: "deadly crash" means others died; a well-known disaster carries its toll.
- The threat is over → score the remaining worry, not the original event.
- Unsure between two numbers → pick the lower.
- Violent deaths (added 25 Oct, from the user's OK on the outlet doubts): civilians or soldiers killed, even a single local shooting = 3; one militant or gang leader killed far away = 2; a murder-suicide where the danger is over = 2; an aftermath or analysis piece about a known attack = 2.
- **Update (25 Oct, the user's rules on the scary-test doubts; these override the line above where they differ):**
  - **A violent death, a shooting, or someone critically wounded by violence = always 3**: militants and suspects included, and one civilian far away included.
  - If the headline states the violent death ("his wife died in the school shooting", "after deadly attack"), it is 3. An aftermath piece that names no death or injury (a suspect's background, a police raid, a policy debate) stays 2.
  - **Accidental deaths can be 2:** a single death, a child included. Several deaths = 3.
  - **Obituaries** for old age or natural death = 0.
- **Update (25 Oct, from the consistency check; defaults that follow the rules above, the user said "ok"):**
  - charges, trials, verdicts or executions over killings, even mass or old ones = 1;
  - an escaped prisoner or fugitive still on the run = 2 (caught = 0);
  - an inquiry, hearing, probe or inquest after an accident = 1 (the death itself as news = 2).
- **Update (27 Oct, the user):** 3 = a death that has already happened, stated as fact (two disease deaths included). Near-death cases that are not a stated death (search called off / presumed dead; explainers about a shooting) sit at 2.5–2.8: label them as integers for now (they round up to 3), and in analysis a score of 2.5 or more counts as 3. Large job cuts stay 2.
