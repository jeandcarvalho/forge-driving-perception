# platform_reference — CarCará code that implements the rules (REFERENCE ONLY)

TypeScript (Fastify + Prisma + MongoDB) copied from the private CarCará platform **unchanged**, except
for three absolute output/input paths (see `../PUBLIC_RELEASE_CHANGES.md`). It is here so a reader can
audit *how* frames were selected, spaced and curated. It is **not runnable** from this repository: it
needs the private MongoDB (`DATABASE_URL`), the Prisma client generated from `prisma/schema.prisma`, and
the original npm workspace (the relative imports assume `apps/api/` and `packages/shared/`).

No front end, authentication, user management, video proxy, LLM-evaluation or visit-counter code is
included.

## Pillar 1 (selection)

- `apps/api/src/controllers/collections/{spacing,add-by-query,split,stats}.ts` — GPS spacing rule
  (greedy, per acquisition, haversine R = 6 371 000 m), query-based collection building, split, stats.
- `apps/api/src/controllers/{dataset-sample,collections/export-manifest}.ts` — sampling and manifest
  export. They build Drive/CDN URLs from file identifiers stored in the database; no identifier is present here.
- `apps/api/src/lib/{build-mongo-match,geo}.ts`, `prisma/schema.prisma`.
- `apps/api/scripts/pillar1-content-export.ts`, `pillar1-spacing-arms.ts` — arm lists / decomposition of the effect.
- `apps/api/scripts/repro-pillar1-uiquery.ts`, `repro-query-signature.ts`, `repro-frame-classes.ts`,
  `repro-corpus-facts*.ts` — attempts to reconstruct the query and corpus facts (read-only).

## Pillar 2 (human curation and second annotator)

- `apps/api/src/controllers/collections/curation/*` — curation state machine, eligibility, statistics, ground-truth resolution.
- `apps/api/scripts/pillar2-select.ts` (deterministic 63-frame selection), `pillar2-create-round2.ts`,
  `pillar2-agreement.ts` (κ, confusion matrix), `audit-curation.ts` (invariants),
  `repro-provenance.ts`, `repro-annotators-313.ts` (annotator identities are anonymized to "Reviewer A/B/…").
- `apps/api/scripts/out/pillar2/agreement-report.json` — the agreement result (public summary).

Not included on purpose: backups of the first annotation round (they contain reviewer accounts),
maintenance/repair scripts, and the web application. The Mongo collection ID of the target collection
appears in several scripts as a constant; it is an identifier, not a credential.
