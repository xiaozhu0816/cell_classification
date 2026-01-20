# LaTeX draft (final crop-5% experiment)

This folder contains a LaTeX draft alongside the Markdown draft.

## Files

- `paper.tex`: main LaTeX manuscript (draft)
- `paper_table_overall_metrics.tex`: LaTeX table generated from `paper_table_overall_metrics.csv`
- `paper_table_temporal_metrics.tex`: LaTeX table generated from `paper_table_temporal_metrics.csv`

The LaTeX draft expects the figures (PNGs) to be in the same folder (they already are for this run).

## Build

If you have a LaTeX distribution installed (e.g., TeX Live or MiKTeX), compile `paper.tex`.

Example (Windows PowerShell, if `pdflatex` is on PATH):

```powershell
pdflatex paper.tex
pdflatex paper.tex
```

(Second pass resolves references.)
