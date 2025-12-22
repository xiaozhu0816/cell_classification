# Report Revision Summary

## Changes Made to EXPERIMENT_REPORT.md

### What Was Wrong in the Original Version:

1. ❌ **No visual evidence:** Results section had tables but didn't reference or include the actual graphs
2. ❌ **Incorrect analysis:** Described numerical summaries without examining the curve patterns
3. ❌ **Missed key patterns:** Didn't identify the critical "performance valley" at 13-19h
4. ❌ **Wrong conclusions:** Claimed simple linear degradation when graphs show complex wave patterns

### What Was Fixed:

1. ✅ **Added graph images:** Copied 4 combined graphs to docs/ folder and embedded in report
2. ✅ **Graph-based analysis:** Completely rewrote results sections focusing on curve patterns
3. ✅ **Identified key patterns:**
   - Performance valley at 13-19h in matched-range experiments
   - Wave patterns vs. flat patterns
   - Test-only line differences between strategies
   - Error bar variations at specific timepoints

4. ✅ **Corrected interpretations:**
   - Full-range: Creates "easy mode" with temporal shortcuts → flat curves
   - Matched-range: Reveals true difficulty → wave patterns with valleys
   - Valley represents biological transitional period, not random noise

### Key Insights Now Properly Documented:

#### From Interval Sweep Graphs:
- **Full-range:** Smooth upward curve, test-only flat at 0.999
- **Matched-range:** Valley at 16h (AUC drops to 0.982), test-only variable
- **Implication:** Temporal shortcuts mask vulnerability

#### From Sliding Window Graphs:
- **Full-range:** Rapid rise then flat plateau at 1.000
- **Matched-range:** Wave pattern with valleys at 7-10h and 16-19h
- **Implication:** Certain infection stages are inherently harder

#### Convergent Evidence:
- Both experiments show same 13-19h vulnerability window
- Both show increased variance at valleys
- Both show flat vs. wavy pattern distinction
- **Confidence:** Systematic effect, not artifact

### Report Structure Now:

```
1. Executive Summary - Updated with graph-based findings
2. Experimental Design - Unchanged (was already good)
3. Interval Sweep Results - COMPLETELY REWRITTEN with graph analysis
4. Sliding Window Results - COMPLETELY REWRITTEN with graph analysis
5. Cross-Experiment Analysis - REWRITTEN to emphasize convergent patterns
6. Discussion - REWRITTEN with biological interpretation of valleys
7. Conclusions - EXPANDED with practical recommendations based on graphs
8. Appendix - Unchanged
```

### Visual Elements Added:

1. `interval_sweep_full_range.png` - Shows smooth curves and flat test-only line
2. `interval_sweep_matched_range.png` - Shows valley at 16h and variable test-only
3. `sliding_window_full_range.png` - Shows plateau pattern
4. `sliding_window_matched_range.png` - Shows wave pattern with multiple valleys

### Main Takeaway:

**Old report:** "Matched sampling reduces performance by 0.5-1%"  
**New report:** "Matched sampling reveals performance valleys at critical infection stages (13-19h) with up to 2% AUC drop, exposing temporal confounding that creates artificially high performance in full-range models"

The new version is evidence-based, visually grounded, and provides actionable insights for deployment.
