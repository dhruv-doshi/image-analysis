# Prompt Version History

Use this file to track prompt iterations and record observations from A/B comparisons
in the "Compare both" mode of the Streamlit app.

## Versions

| Version | File | Created | Status | Description |
|---------|------|---------|--------|-------------|
| v1 | `system.md` | 2026-02-28 | active | Balanced critic — tone-matched to quality tier; leads with strengths for good/excellent images |
| v2 | `system_v2.md` | 2026-03-03 | testing | Flaw-focused critic — always leads with mistakes, minimises positives, emphasises fixes |

## How to promote a winner

1. Copy the winning file over `system.md` (this is what the app uses by default).
2. Keep the old file as `system_vN.md` for reference.
3. Update the Status column above (active / archived / testing).
4. Add a note in the Testing Log below.

## Testing Log

Add observations here after running side-by-side comparisons.

<!--
Template:
### YYYY-MM-DD — [image description]
- **v1**: [what it said / how it felt]
- **v2**: [what it said / how it felt]
- **Verdict**: [which was better and why]
-->
