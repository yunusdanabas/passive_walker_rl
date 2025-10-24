# Overnight PPO Training Instructions

**Created:** October 24, 2025  
**Estimated Total Time:** 8-11 hours

---

## Quick Start

### Option 1: Run in Background (Recommended)
```bash
cd /home/yunusdanabas/passive_walker_rl
nohup ./run_overnight_ppo_training.sh > overnight_training.log 2>&1 &
```

**This will:**
- Run all 3 training configurations sequentially
- Save main output to `overnight_training.log`
- Save detailed logs per run in `experiments/ppo_runs/overnight_logs/`
- Continue running even if you log out
- Complete in 8-11 hours

### Option 2: Run in Foreground (Terminal Must Stay Open)
```bash
cd /home/yunusdanabas/passive_walker_rl
./run_overnight_ppo_training.sh
```

---

## What Gets Trained

### Run A: MLP Baseline (~1-2 hours)
- **Model:** MLP [64, 64]
- **Timesteps:** 500,000
- **Features:** Basic PPO
- **Purpose:** Fast baseline

### Run B: LSTM + Curriculum (~3-4 hours)
- **Model:** LSTM 128x2
- **Timesteps:** 1,000,000
- **Features:** Curriculum learning
- **Purpose:** Temporal modeling

### Run C: LSTM + Full Enhancement (~4-5 hours)
- **Model:** LSTM 128x2
- **Timesteps:** 1,000,000
- **Features:** Curriculum + Aggressive domain randomization
- **Purpose:** Maximum robustness

---

## Monitoring Progress

### Check if Running
```bash
ps aux | grep run_overnight_ppo_training
```

### View Main Log (Real-time)
```bash
tail -f overnight_training.log
```

### View Specific Run Log
```bash
tail -f experiments/ppo_runs/overnight_logs/ppo_mlp_baseline_*.log
tail -f experiments/ppo_runs/overnight_logs/ppo_lstm_curriculum_*.log
tail -f experiments/ppo_runs/overnight_logs/ppo_lstm_advanced_*.log
```

### View TensorBoard (While Training)
```bash
tensorboard --logdir=experiments/ppo_runs
# Open browser to http://localhost:6006
```

---

## Expected Output Structure

```
experiments/ppo_runs/
├── overnight_logs/
│   ├── ppo_mlp_baseline_YYYYMMDD_HHMMSS.log
│   ├── ppo_lstm_curriculum_YYYYMMDD_HHMMSS.log
│   ├── ppo_lstm_advanced_YYYYMMDD_HHMMSS.log
│   └── overnight_summary_YYYYMMDD_HHMMSS.txt
├── ppo_mlp_baseline/
│   ├── final_model.pth
│   ├── config.json
│   └── events.out.tfevents.*
├── ppo_lstm_curriculum/
│   ├── final_model.pth
│   ├── config.json
│   └── events.out.tfevents.*
└── ppo_lstm_advanced/
    ├── final_model.pth
    ├── config.json
    └── events.out.tfevents.*
```

---

## Progress Indicators

The script prints regular updates:
```
[2025-10-24 23:00:00] Starting: ppo_mlp_baseline
[2025-10-24 23:00:01] Log file: experiments/ppo_runs/overnight_logs/...
Starting PPO training: ppo_mlp_baseline
Timestep 10000: Avg return = 1.23, Avg length = 245.00
Timestep 20000: Eval return = 2.45 ± 0.32
...
[2025-10-25 00:30:00] ✅ Completed: ppo_mlp_baseline
```

---

## Stopping Training

### Graceful Stop (Saves Progress)
```bash
# Find the process ID
ps aux | grep run_overnight_ppo_training
# Send interrupt signal (Ctrl+C equivalent)
kill -INT <PID>
```

### Force Stop (May Lose Current Run)
```bash
pkill -f run_overnight_ppo_training
```

---

## After Training Completes

### 1. Check Summary
```bash
cat experiments/ppo_runs/overnight_logs/overnight_summary_*.txt
```

### 2. View Training Curves
```bash
tensorboard --logdir=experiments/ppo_runs
```

### 3. Evaluate Models
```bash
# Evaluate Run A
python -m passive_walker.ppo.evaluate_cli \
  --model experiments/ppo_runs/ppo_mlp_baseline/final_model.pth \
  --n_eval_episodes 20

# Evaluate Run B
python -m passive_walker.ppo.evaluate_cli \
  --model experiments/ppo_runs/ppo_lstm_curriculum/final_model.pth \
  --n_eval_episodes 20

# Evaluate Run C
python -m passive_walker.ppo.evaluate_cli \
  --model experiments/ppo_runs/ppo_lstm_advanced/final_model.pth \
  --n_eval_episodes 20
```

---

## Troubleshooting

### Script Won't Run
```bash
# Make sure it's executable
chmod +x run_overnight_ppo_training.sh

# Check Python environment
mamba activate main
python --version
```

### Out of Memory
- The script uses CPU by default (safer for long runs)
- If needed, edit script to reduce batch_size or n_envs
- Each run is independent, can comment out some runs

### Training Crashes
- Check individual run logs in `experiments/ppo_runs/overnight_logs/`
- Script continues with remaining runs if one fails
- Can restart failed runs individually

---

## Manual Run (If Script Fails)

Run each configuration separately:

```bash
# Run A
python -m passive_walker.ppo.train \
  --experiment_name ppo_mlp_baseline \
  --model_type mlp --timesteps 500000 \
  --out experiments/ppo_runs

# Run B
python -m passive_walker.ppo.train \
  --experiment_name ppo_lstm_curriculum \
  --model_type lstm --hidden_size 128 --num_layers 2 \
  --use_curriculum --timesteps 1000000 \
  --out experiments/ppo_runs

# Run C
python -m passive_walker.ppo.train \
  --experiment_name ppo_lstm_advanced \
  --model_type lstm --hidden_size 128 --num_layers 2 \
  --use_curriculum --use_domain_randomization \
  --randomization_profile aggressive \
  --timesteps 1000000 --out experiments/ppo_runs
```

---

## Expected Timeline (Example)

| Time | Status |
|------|--------|
| 23:00 | Script started |
| 23:05 | Run A (MLP) training begins |
| 00:30 | Run A completes, Run B (LSTM+Curriculum) begins |
| 04:00 | Run B completes, Run C (LSTM+Full) begins |
| 08:30 | Run C completes, script finishes |
| 08:31 | Summary generated, all done! |

**Total: ~9.5 hours**

---

## Notes

- Script runs sequentially (one at a time) to avoid resource conflicts
- Each run saves checkpoints every 50k timesteps
- Evaluation runs every 10k timesteps
- All logs are timestamped for easy identification
- Failed runs don't stop subsequent runs
- Summary file generated at completion

---

## Ready to Start!

**Recommended command:**
```bash
cd /home/yunusdanabas/passive_walker_rl
nohup ./run_overnight_ppo_training.sh > overnight_training.log 2>&1 &
echo "Training started! PID: $!"
```

**Then check progress:**
```bash
tail -f overnight_training.log
```

**Or detach and check tomorrow!** ☕

