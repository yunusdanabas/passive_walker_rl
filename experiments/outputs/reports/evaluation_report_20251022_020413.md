# BC Model Evaluation Report

**Generated:** 2025-10-22 02:04:13

## Executive Summary

| Model | Overall Success Rate | Avg Reward | Best Condition | Worst Condition |
|-------|---------------------|------------|----------------|-----------------|
| Hip Control | 83.3% | 277.04 | nominal | steep |
| Both Joints | 83.3% | 277.04 | nominal | steep |

## Detailed Results

### Hip Control

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

### Both Joints

| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |
|-----------|--------------|------------|-----------|--------------|
| nominal | 100.0% | 322.53 | 2000.0 | 20.0s |
| gentle | 100.0% | 319.75 | 2000.0 | 20.0s |
| steep | 0.0% | 56.26 | 329.0 | 3.3s |
| low_friction | 100.0% | 322.03 | 2000.0 | 20.0s |
| high_friction | 100.0% | 321.61 | 2000.0 | 20.0s |
| gentle_low | 100.0% | 320.07 | 2000.0 | 20.0s |

## Analysis

**Best Overall Model:** Hip Control

### Condition Analysis

| Condition | Best Model | Success Rate | Notes |
|-----------|------------|--------------|-------|
| nominal | Hip Control | 100.0% | Easy condition |
| gentle | Hip Control | 100.0% | Easy condition |
| steep | Hip Control | 0.0% | Challenging condition |
| low_friction | Hip Control | 100.0% | Easy condition |
| high_friction | Hip Control | 100.0% | Easy condition |
| gentle_low | Hip Control | 100.0% | Easy condition |

## Recommendations

1. **For PPO Training:** Use the best performing model as initialization
2. **For Robustness:** Focus on conditions where models struggle
3. **For Data Collection:** Collect more data for challenging conditions
4. **For Evaluation:** Use comprehensive physics condition testing

