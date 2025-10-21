# BC Model Evaluation Report

**Generated:** 2025-10-22 00:44:51

## Executive Summary

| Model | Overall Success Rate | Avg Reward | Best Condition | Worst Condition |
|-------|---------------------|------------|----------------|-----------------|
| Baseline | 83.3% | 277.04 | nominal | steep |
| Enhanced | 83.3% | 277.04 | nominal | steep |
| Gentle | 83.3% | 277.04 | nominal | steep |
| Low_Friction | 83.3% | 277.04 | nominal | steep |
| Mass_Jitter | 83.3% | 277.04 | nominal | steep |

## Detailed Results

### Baseline

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

### Enhanced

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

### Gentle

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

### Low_Friction

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

### Mass_Jitter

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

## Analysis

**Best Overall Model:** Baseline

### Condition Analysis

| Condition | Best Model | Success Rate | Notes |
|-----------|------------|--------------|-------|
| nominal | Baseline | 100.0% | Easy condition |
| gentle | Baseline | 100.0% | Easy condition |
| steep | Baseline | 0.0% | Challenging condition |
| low_friction | Baseline | 100.0% | Easy condition |
| high_friction | Baseline | 100.0% | Easy condition |
| gentle_low | Baseline | 100.0% | Easy condition |

## Recommendations

1. **For PPO Training:** Use the best performing model as initialization
2. **For Robustness:** Focus on conditions where models struggle
3. **For Data Collection:** Collect more data for challenging conditions
4. **For Evaluation:** Use comprehensive physics condition testing

