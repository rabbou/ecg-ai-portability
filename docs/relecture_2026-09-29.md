# Reading pass of the public repository, 29 September 2026

Proposed GitHub description, not applied:

> A 12-lead ECG threshold calibrated at one hospital and used unchanged at others, measured diagnosis by diagnosis.

The live description reads "Does a conformal coverage guarantee survive a change of hospital? Measured on 12-lead ECG across three public corpora." It carries the phrase rejected on 27/9, asks a question built on a technical term, and counts three corpora where the study reads seven.

Scope: README.md, REPORT.md, QUESTIONS.md, PLAN.md, CITATION.cff, pyproject.toml, mappings/NOTICE.md, third_party/ecgfounder/PROVENANCE.md, the figure text drawn by scripts/figures.py, and the GitHub description. Each is at most two clicks from the repository page. Measured against the `writing` charter, its two rules of 27/9 (a study title names the idea in the reader's words; a public copy never denies a suspicion nobody raised), and rows 6 and 23 of the site audit. Line numbers are those of 5e6835f, before any fix.

"Fixed" means changed in this branch. "Open" names the reason the change is not made here.

| # | Where | Problem | Fix | Status |
|---|---|---|---|---|
| G1 | GitHub description | rejected phrase, question built on a technical term, "three corpora" where the study reads seven | the line at the top of this file | open: a public change, applied by Ruben |
| P1 | pyproject.toml:4 | the same sentence as the live description | the proposed line | fixed |
| T1 | REPORT.md:1, CITATION.cff:3 | the title names the method, "Per-label conformal calibration of an ECG classifier across hospitals" (charter rule 3) | proposed: "An ECG threshold carried to another hospital, measured label by label" | open: the title is the citation record and the name the site page and outreach quote, so it changes with them |
| T2 | REPORT.md:5 | "draft of 9 September 2026", while CITATION.cff records a release on 2026-09-08 and section 3.6 was written after both dates | Ruben states whether the public text is a draft, and its date | open: a fact only Ruben holds |
| R1 | README.md:3-7 | opens on the method and on "the 90% coverage guarantee" (rules 2 and 3) | opens on the finding and its four numbers (outcomes.json, target_scale.json), method second | fixed |
| R2 | README.md:7, 12, 172 | "break" as shorthand for a change of hospital | "change of hospital", "result" | fixed |
| R3 | README.md:14 | "the three figures" while the report shows five | "their figures" | fixed |
| R4 | README.md:16 | "This file documents..." (the document describes itself, lint) | "What follows covers..." | fixed |
| R5 | README.md:42-49 | change narrative: "did not keep those copies together", "is now one splitting unit" | present tense | fixed |
| R6 | README.md:117 | "The guarantee is asymptotic" | "Its coverage holds only asymptotically" | fixed |
| R7 | README.md:163-170 | two paragraphs say the same thing; "(C-12)" is a plan criterion id a reader cannot use | one paragraph, id removed | fixed |
| R8 | README.md:190 | "redraws all six figures" while the script draws eight | "every figure", and the French command beside it | fixed |
| R9 | README.md:236 | em dash in a README (budget zero), "+" as notation | a sentence | fixed |
| R10 | README.md:239-243 | heading "What this is not" (named tic) and a chain of negated items (lint) | "Scope", one clause per fact | fixed |
| R11 | README.md:275 | two em dashes around an aside | two sentences | fixed |
| Q1 | QUESTIONS.md:3 | the opener describes the document | opens on where each number comes from | fixed |
| Q2 | QUESTIONS.md:7 | the clinician answer rests on "guarantee" and "promise"; its question is the one Ruben's 27/9 example comes from | the 90% said as an average over patients, no replacement word needed | fixed |
| Q3 | QUESTIONS.md:15 | "what calibration promises survive" | "whether calibrated coverage survives" | fixed |
| Q4 | QUESTIONS.md:19 | "harness" as jargon; "a way to measure the promise, not an assurance" (negative parallelism) | "the measurement code"; what a site can and cannot learn | fixed |
| Q5 | QUESTIONS.md:23 | sets 73.2% (outcomes.json, whose marginal is 89.9%) beside "90.0%" (shift.json, whose MI figure is 73.5%) in the same sentence, where REPORT.md says two results files never share one | 89.9% would match its own file | open: a number, reported to Ruben and not changed |
| Q6 | QUESTIONS.md:23 | links a clinician to PLAN.md, a dated execution plan, for criterion C-11 | see PL1 | open: follows PL1 |
| Q7 | QUESTIONS.md:27 | "Its guarantee is asymptotic"; "paired" said twice in one clause | "Its coverage holds only asymptotically"; one "paired" | fixed |
| Q8 | QUESTIONS.md:37 | change narrative: "is now drawn", "what an earlier version of the file got wrong" | present tense | fixed |
| Q9 | QUESTIONS.md:41 | chain of four negated items (lint) | one clause per fact | fixed |
| Q10 | QUESTIONS.md:45 | bare "guarantee" four times in the technical answer on calibrated probabilities | kept as the technical term | open: see K1 |
| Q11 | QUESTIONS.md:49 | "the generality bullet" names a bullet head that T21 removes | "keeps them among the limitations" | fixed |
| Q12 | QUESTIONS.md:59, 63 | heading "What was the duplicate leak changing?" reads badly; "is now" | "What did the duplicated tracings change?"; present tense | fixed |
| T3 | REPORT.md:13 | chain of two negations (lint) | two plain clauses | fixed |
| T4 | REPORT.md:27 | "satisfies its stated guarantee" | "meets its stated target" | fixed |
| T5 | REPORT.md:41 | "The other is a reader study" with no first study named as one of two | "The second ECG study" | fixed |
| T6 | REPORT.md:45 | "This report addresses both" (lint) and an aphoristic close | "Both are measured here, with established techniques throughout." | fixed |
| T7 | REPORT.md:79 | change narrative about the archived predictions (lint) | "link to another study's files" | fixed |
| T8 | REPORT.md:87 | chain of three negations (lint) | "unweighted, neither resampled nor augmented" | fixed |
| T9 | REPORT.md:100 | author comment on the text: "part of the answer and not an aside" | the reason stated once | fixed |
| T10 | REPORT.md:126 | caption says coral; the figure draws MI in red (#b4423a) | "Red" | fixed |
| T11 | REPORT.md:130 | "The guarantee was at no point violated" | "The 90% target was met throughout" | fixed |
| T12 | REPORT.md:153 | caption names the third column twice; the figure titles the middle column "Mondrian" where the text says class-conditional | caption and figure both say pooled and class-conditional | fixed |
| T13 | REPORT.md:155 | aphoristic close on "an asymptotic guarantee" | a plain comparison of the two corrections | fixed |
| T14 | REPORT.md:161 | "The guarantee class-conditional calibration delivers is conditional on the label" | "holds coverage within each label, not within a subgroup of one" | fixed |
| T15 | REPORT.md:195, 248, 279 | change narrative three times (lint) | "cannot", "does not", "does not" | fixed |
| T16 | REPORT.md:229 | "A coverage guarantee stated over a whole population" | "Coverage met over a whole population" | fixed |
| T17 | REPORT.md:231 | "What that gap is not is evidence against..." | "That gap does not count against..." | fixed |
| T18 | REPORT.md:233 | "rather than" twice in one clause (lint); "let the formalism take credit for the arithmetic" (mannered) | one contrast; "credit the per-label thresholds, not the formalism" | fixed |
| T19 | REPORT.md:235 | "coverage guarantees fail under domain and label shift" | "conformal coverage falls below its target" | fixed |
| T20 | REPORT.md:239-251 | thirteen bullets opening on a bold head followed by a gloss (named tic), among them "Marginal guarantee" | heads dropped; each bullet's first sentence already states the limit | fixed |
| T21 | REPORT.md:245 | "the class-conditional guarantee" | "class-conditional coverage" | fixed |
| T22 | REPORT.md:251 | chain of four negations (lint) | one clause per fact | fixed |
| T23 | REPORT.md:266 | "coverage guarantee" inside the quotation of Mehrtens et al. [12] | none | open: a quotation stays verbatim |
| T24 | REPORT.md:279 | "it now records the seed..., which an earlier version of it did not" | "it records the seed" | fixed |
| K1 | REPORT.md:19 (twice), 35 (twice), 37, 39 (twice), 100; QUESTIONS.md:45 (four times) | bare "guarantee" where the text defines what conformal prediction proves | kept: these name the theorem, and none needed a replacement word; no `[COUVERTURE: à choisir]` was written | open: if the word chosen today also replaces the bare term, these are the spots |
| F1 | scripts/figures.py:525 | "--" standing for a dash in the three threshold panel titles (audit row 23) | an en dash | fixed |
| F2 | scripts/figures.py:86-93 | "drawn by scripts/figures.py from results/..." footer on all eight figures (audit row 23) | removed | fixed |
| F3 | scripts/figures.py:614-616 | the outcomes note typed "~10%" and "27%" | computed from outcomes.json; the same two values print | fixed |
| F4 | audit row 6 | the three figures on the French page are English-only | `--lang fr` draws fig1_thresholds_fr, fig2_outcomes_fr, fig3_coverage_fr from the same functions | fixed here; the site copies them in its own repository |
| F5 | scripts/figures.py:1-31 | docstring says "six figures", lists four, uses "--" for dashes | a table of the eight figures | fixed |
| F6 | scripts/figures.py:307-311 | docstring says "four encoder arms" (there are five) and "a guarantee that stopped holding" | "five", "coverage lost at that hospital" | fixed |
| C1 | src/ecs/__init__.py:1, src/ecs/report.py:201, tests/test_conformal.py:1 | the rejected phrase in a docstring or comment | reworded | fixed |
| C2 | scripts/shift_table.py:1, 347; scripts/arms_table.py:3, 388; src/ecs/splits.py:3 | the rejected phrase; lines 347 and 388 write it into the `question` field of results/shift.json and results/arms.json | none here | open: these sources are digested by the results files they build, so one changed byte fails tests/test_provenance.py, and rewording the JSON means rebuilding results, outside a wording pass |
| N1 | mappings/NOTICE.md:18 | a pair of em dashes in a repository document | commas | fixed |
| PL1 | PLAN.md | a dated execution plan ("Six working days: Mon 24 → Fri 28 Aug") in the public root, linked from QUESTIONS.md; the linter passes it | none here | open: whether a public repository keeps its plan is Ruben's call |

Read and found clean: third_party/ecgfounder/PROVENANCE.md, LICENSE, the CITATION.cff abstract.

This review was done in the build session, which the brief forbade from starting a sub-agent. The fresh-context pass the charter's publication sequence asks for has not run on this text.
