# med-dx corpus pipeline

Builds the corpus the `med-dx` benchmark reads. `scripts/prepare_med_dx.py` installs a built corpus into the data root; point
it at one with `--corpus` or `MED_DX_CORPUS`.

Rebuilding is not free — synthesis is two LLM calls per case over 300 cases —
and it will not reproduce byte-identically, because this endpoint rejects
`temperature=0` and EviDx's synthesiser asks for it. The cached files *are* the
environment.

## Order

    data/fetch.py           the three source datasets from HF
    data/fetch_evidence.py  Med-Evidence-2.6k (case text, gold, evidence spans)
    build_corpus.py         join options, sample 100/split, emit corpus-*.jsonl
    classify_focus.py       label what each case actually asks
    synthesize.py           EviDx's synthesiser + leakage guard -> ehr-*.jsonl

## Audits

Each of these justified a change.

    check_fidelity.py       evidence survival (99.2%) and answer leakage
    check_contamination.py  what synthesis asserted that the source did not
    env.py                  inspect one case's environment by hand

## Why it is not upstream's pipeline unchanged

Two departures, both measured:

- A **leakage guard** on the synthesis prompts. Without it, 13 of 30 records
  stated a diagnosis in a section visible before any tool call, inferred rather
  than copied. EviDx's prompts have no rule against naming the answer.
- **Answer options stripped from JAMA case text.** 34 of 100 JAMA cases had
  option strings running into the narrative with no delimiter, and the
  synthesiser filed them under `other_procedures` — so the record stated that
  candidate answers had already been performed.

The cost is that the environment is ours, so numbers here do not belong in
EviDx's published table.
