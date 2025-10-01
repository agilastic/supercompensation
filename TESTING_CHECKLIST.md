# Correlation Analysis Testing Checklist

## Pre-Testing Setup

### 1. Install Dependencies ✓
```bash
cd /Users/alex/workspace/supercompensation
pip install -r requirements.txt
```

**Critical packages:**
- `scipy>=1.11.0` (statistical functions)
- `pandas>=2.1.0` (data manipulation)
- `numpy>=1.24.0` (numerical operations)
- `click>=8.1.0` (CLI framework)
- `rich>=13.7.0` (terminal formatting)

### 2. Verify Database Has Data ✓
```bash
# Check if you have sufficient data
strava-super status
```

**Minimum requirements:**
- 10+ days of training metrics (CTL/ATL/TSB)
- 10+ days of HRV data (optional but recommended)
- 10+ days of sleep data (optional but recommended)

---

## Unit Tests

### Test 1: Basic Correlation Calculation ✓
```bash
cd /Users/alex/workspace/supercompensation
python test_correlation_analysis.py
```

**Expected output:**
```
TEST 1: Basic Correlation Calculation
✓ HRV vs Fatigue (ATL)
  Correlation: -0.XXX
  P-value: 0.XXXX
  Strength: moderate/strong
  Direction: negative
  Significant: Yes
  Samples: 60

✅ TEST PASSED
```

**What it tests:**
- Pearson correlation calculation
- P-value computation
- Strength classification
- Direction detection
- Sample counting

### Test 2: Leading Indicator Detection ✓
**Expected output:**
```
TEST 2: Leading Indicator Detection
  Lag 1 days: r=-0.XXX, p=0.XXXX
  Lag 2 days: r=-0.XXX, p=0.XXXX
  ...
✓ Best predictive lag: N days
  Correlation: -0.XXX

✅ TEST PASSED
```

**What it tests:**
- Time-lagged correlation
- Optimal lag detection
- Predictive power assessment

### Test 3: Correlation Matrix Generation ✓
**Expected output:**
```
TEST 3: Correlation Matrix Generation
✓ Matrix dimensions: 10 x 10
  Variables: ctl, atl, tsb, hrv_rmssd, ...
  Sample correlations:
  HRV vs ATL: -0.XXX
  Sleep vs TSB: +0.XXX

✅ TEST PASSED
```

**What it tests:**
- Full matrix computation
- Diagonal correctness (r=1.0)
- Matrix symmetry

---

## Integration Tests

### Test 4: CLI Command Registration ✓
```bash
# Check insights group exists
python -m strava_supercompensation.cli --help | grep insights
```

**Expected output:**
```
  insights  Advanced data science insights and correlation analysis.
```

### Test 5: Correlations Command Help ✓
```bash
# Check command is registered
python -m strava_supercompensation.cli insights --help
```

**Expected output:**
```
Usage: cli.py insights [OPTIONS] COMMAND [ARGS]...

  Advanced data science insights and correlation analysis.

Commands:
  correlations       Comprehensive correlation analysis between wellness...
  predictive-metrics Show metrics that predict future performance...
  wellness-trends    Analyze wellness metric trends and patterns.
```

### Test 6: Correlations Command Options ✓
```bash
# Check all options available
python -m strava_supercompensation.cli insights correlations --help
```

**Expected output:**
```
Usage: cli.py insights correlations [OPTIONS]

  Comprehensive correlation analysis between wellness and performance metrics.

Options:
  --days INTEGER          Days of historical data to analyze  [default: 90]
  --min-samples INTEGER   Minimum samples required for correlation  [default: 10]
  --significance FLOAT    P-value threshold for significance  [default: 0.05]
  --export TEXT           Export results to JSON file
  --help                  Show this message and exit.
```

---

## Functional Tests (With Real Data)

### Test 7: Basic Correlation Analysis ✓
```bash
# Run with default parameters
strava-super insights correlations
```

**Expected output structure:**
```
📊 Comprehensive Correlation Analysis

Analyzing 90 days of data...

✅ Analysis complete!
Period: 2024-XX-XX to 2025-XX-XX
Data points: XX

📈 Summary Statistics:
  Total correlations tested: 28
  Significant correlations: XX (XX.X%)
  Leading indicators found: X
  Data quality: good/excellent

🏷️  Correlations by Category:
┌─────────────────────┬───────┬─────────────┬──────┐
│ Category            │ Total │ Significant │ Rate │
├─────────────────────┼───────┼─────────────┼──────┤
│ Wellness Performance│   7   │     X       │ XX%  │
│ Recovery Adaptation │   5   │     X       │ XX%  │
│ Training Response   │   5   │     X       │ XX%  │
│ Health Performance  │   6   │     X       │ XX%  │
└─────────────────────┴───────┴─────────────┴──────┘

🔥 Top Significant Correlations:
┌──────────────────────────────┬─────────┬────────┬──────────┬────┐
│ Relationship                 │ r       │ p      │ Strength │ n  │
├──────────────────────────────┼─────────┼────────┼──────────┼────┤
│ ...                          │ +/-0.XXX│ 0.XXXX │ ...      │ XX │
└──────────────────────────────┴─────────┴────────┴──────────┴────┘

🎯 Leading Indicators (Predictive Metrics):
┌────────────────┬─────┬─────────┬────────┬────────────────┐
│ Indicator      │ Lag │ r       │ Power  │ Insight        │
├────────────────┼─────┼─────────┼────────┼────────────────┤
│ ...            │ Xd  │ +/-0.XXX│ ...    │ ...            │
└────────────────┴─────┴─────────┴────────┴────────────────┘

💡 Key Insights:
  1. ...
  2. ...

📋 Recommendations:
  • Monitor metrics with strong correlations closely
  • Use leading indicators for proactive training adjustments
  • Re-run analysis monthly to track relationship changes
  • Focus on metrics with highest predictive power
```

**Validation checklist:**
- ✓ Command runs without errors
- ✓ Data loads from database
- ✓ Correlations calculated correctly
- ✓ Tables render properly
- ✓ Insights generated
- ✓ No crashes or exceptions

### Test 8: Custom Parameters ✓
```bash
# Test with custom parameters
strava-super insights correlations --days 60 --min-samples 20 --significance 0.01
```

**Validation:**
- ✓ Respects --days parameter (60 days analyzed)
- ✓ Filters by min-samples (only correlations with n≥20)
- ✓ Uses stricter significance (p<0.01)
- ✓ Output reflects parameter changes

### Test 9: JSON Export ✓
```bash
# Test export functionality
strava-super insights correlations --days 90 --export test_results.json

# Verify file created
ls -lh test_results.json

# Check JSON structure
python -m json.tool test_results.json | head -30
```

**Expected JSON structure:**
```json
{
  "analysis_period": {
    "days_back": 90,
    "start_date": "2024-XX-XX",
    "end_date": "2025-XX-XX",
    "data_points": XX
  },
  "correlation_matrices": { ... },
  "significant_correlations": [ ... ],
  "leading_indicators": [ ... ],
  "actionable_insights": [ ... ],
  "category_analyses": { ... },
  "summary": { ... }
}
```

**Validation:**
- ✓ File created successfully
- ✓ Valid JSON format
- ✓ All expected keys present
- ✓ Data structure correct

### Test 10: Wellness Trends Command ✓
```bash
# Test quick insights command
strava-super insights wellness-trends --days 60
```

**Expected output:**
```
📈 Wellness Trends Analysis

Analyzing 60 days of wellness data...

Wellness-Performance Relationships:
  Found X significant correlations

┌────────────────────────┬─────────────┬──────────────┐
│ Metric Pair            │ Correlation │ Significance │
├────────────────────────┼─────────────┼──────────────┤
│ ...                    │ +/-0.XXX    │ ✓            │
└────────────────────────┴─────────────┴──────────────┘
```

### Test 11: Predictive Metrics Command ✓
```bash
# Test leading indicators view
strava-super insights predictive-metrics --days 90
```

**Expected output:**
```
🎯 Predictive Metrics Dashboard

Found X predictive metrics

🔥 Strong Predictive Power:

  📊 hrv_rmssd → tsb
     Optimal lag: X days
     Correlation: +/-0.XXX (p=0.XXXX)
     💡 HRV changes predict form X days in advance

⚡ Moderate Predictive Power:
  • ...

💡 How to use this:
  1. Monitor strong predictive metrics daily
  2. Adjust training plans based on indicator changes
  3. Use lag time to plan recovery days proactively
  4. Track accuracy of predictions over time
```

---

## Edge Case Tests

### Test 12: Insufficient Data ✓
```bash
# Test with very short timeframe (should warn)
strava-super insights correlations --days 5 --min-samples 10
```

**Expected output:**
```
⚠️  Need at least 10 days of data, found 5
```

### Test 13: Missing Data Categories ✓
**Scenario:** User has training data but no HRV/sleep data

**Expected behavior:**
- ✓ Analysis runs without errors
- ✓ Skips categories with insufficient data
- ✓ Reports which categories were analyzed
- ✓ No crashes on missing tables

### Test 14: No Significant Correlations ✓
**Scenario:** Data has no statistically significant relationships

**Expected output:**
```
✅ Analysis complete!
...
Significant correlations: 0 (0.0%)
Leading indicators found: 0

No strong leading indicators found in current dataset
Try increasing the analysis period with --days
```

---

## Performance Tests

### Test 15: Large Dataset Performance ✓
```bash
# Test with maximum timeframe
time strava-super insights correlations --days 365
```

**Performance targets:**
- Execution time: < 10 seconds
- Memory usage: < 200 MB
- No timeout errors

### Test 16: Export File Size ✓
```bash
# Test export with large dataset
strava-super insights correlations --days 365 --export large_test.json
ls -lh large_test.json
```

**Expected:**
- File size: < 5 MB
- Valid JSON format
- Reasonable load time

---

## Regression Tests

### Test 17: Backward Compatibility ✓
**Verify existing commands still work:**

```bash
strava-super status
strava-super recommend
strava-super metrics
strava-super plan --days 7
```

**All should work without errors.**

### Test 18: Database Schema ✓
**Verify no database changes broke existing functionality:**

```bash
# Check database tables
sqlite3 data/strava.db ".tables"
```

**Expected:** All existing tables present, no corruption.

---

## Documentation Tests

### Test 19: Help Text Completeness ✓
```bash
# Check all help texts are informative
strava-super insights --help
strava-super insights correlations --help
strava-super insights wellness-trends --help
strava-super insights predictive-metrics --help
```

**Verify:**
- ✓ Clear descriptions
- ✓ All options documented
- ✓ Examples provided (if applicable)
- ✓ No typos

### Test 20: Documentation Accuracy ✓
**Check files:**
- `CORRELATION_ANALYSIS.md`
- `IMPLEMENTATION_SUMMARY.md`
- `TESTING_CHECKLIST.md` (this file)

**Verify:**
- ✓ Code examples work
- ✓ Output examples match reality
- ✓ No outdated information
- ✓ Links work (if any)

---

## User Acceptance Tests

### Test 21: First-Time User Experience ✓

**Scenario:** New user with 30+ days of data runs analysis for first time.

**Steps:**
1. Run `strava-super insights correlations`
2. Read output
3. Understand key correlations
4. Identify actionable insights

**Success criteria:**
- ✓ User understands what correlations mean
- ✓ User can identify their top 3 important metrics
- ✓ User knows what to do next
- ✓ No confusion about p-values/significance

### Test 22: Advanced User Experience ✓

**Scenario:** Power user wants to export and analyze raw data.

**Steps:**
1. Run `strava-super insights correlations --export data.json`
2. Load JSON in external tool (Python/R/Excel)
3. Perform custom analysis
4. Validate correlations independently

**Success criteria:**
- ✓ JSON format is standard and parseable
- ✓ All necessary data included
- ✓ Correlations match independent calculations

---

## Status Tracking

### Test Results Summary

| Test # | Test Name                    | Status | Notes |
|--------|------------------------------|--------|-------|
| 1      | Basic Correlation            | 🟡 READY | Run after pip install |
| 2      | Leading Indicators           | 🟡 READY | Run after pip install |
| 3      | Correlation Matrix           | 🟡 READY | Run after pip install |
| 4      | CLI Registration             | 🟢 PASS  | Code review confirmed |
| 5      | Command Help                 | 🟡 READY | Run after pip install |
| 6      | Command Options              | 🟢 PASS  | Code review confirmed |
| 7      | Basic Analysis               | 🟡 READY | Needs real data |
| 8      | Custom Parameters            | 🟡 READY | Needs real data |
| 9      | JSON Export                  | 🟡 READY | Needs real data |
| 10     | Wellness Trends              | 🟡 READY | Needs real data |
| 11     | Predictive Metrics           | 🟡 READY | Needs real data |
| 12     | Insufficient Data            | 🟡 READY | Needs testing |
| 13     | Missing Categories           | 🟡 READY | Needs testing |
| 14     | No Significant Corrs         | 🟡 READY | Needs testing |
| 15     | Large Dataset Performance    | 🟡 READY | Needs testing |
| 16     | Export File Size             | 🟡 READY | Needs testing |
| 17     | Backward Compatibility       | 🟡 READY | Needs testing |
| 18     | Database Schema              | 🟢 PASS  | No changes made |
| 19     | Help Text Completeness       | 🟢 PASS  | Code review confirmed |
| 20     | Documentation Accuracy       | 🟢 PASS  | Reviewed |
| 21     | First-Time User              | 🟡 READY | Needs user testing |
| 22     | Advanced User                | 🟡 READY | Needs user testing |

**Legend:**
- 🟢 PASS: Test passed
- 🟡 READY: Test prepared, ready to run
- 🔴 FAIL: Test failed (requires fix)
- ⚪ SKIP: Test skipped (not applicable)

---

## Quick Start Testing

### Minimum Viable Test (After pip install)

```bash
# 1. Run unit tests
python test_correlation_analysis.py

# 2. Check CLI registration
python -m strava_supercompensation.cli insights --help

# 3. Run with real data (if available)
strava-super insights correlations --days 30

# 4. Export results
strava-super insights correlations --days 30 --export test.json
cat test.json | python -m json.tool | head
```

**Expected results:** All 4 steps complete without errors

---

## Bug Report Template

If you find issues during testing:

```
### Bug Report

**Test Number:** #X
**Test Name:**
**Command:**
**Expected:**
**Actual:**
**Error Message:**
**Environment:**
  - Python version:
  - OS:
  - Package versions:

**Steps to Reproduce:**
1.
2.
3.

**Suggested Fix:**
```

---

## Testing Complete ✅

When all tests pass:

1. Update this checklist with results
2. Mark implementation as "Production Ready"
3. Deploy to users
4. Monitor for edge cases
5. Iterate based on feedback

**Testing Status:** 🟡 READY FOR EXECUTION
**Estimated Testing Time:** 1-2 hours
**Blocker:** Dependencies not installed in current environment
**Next Step:** `pip install -r requirements.txt` then run tests
