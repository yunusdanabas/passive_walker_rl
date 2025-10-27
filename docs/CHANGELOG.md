# Changelog

## Unreleased

### Major Reorganization (Current)

- **Documentation**: Reduced from 25+ files to 5 essential documents (README, SETUP, TRAINING, API, CHANGELOG)
- **Experiments**: Unified all models and results into `experiments/` directory structure
- **BC Module**: Archived advanced features (ensemble, multi-task learning, analysis tools)
- **PPO Module**: Simplified to basic trainer only (archived enhanced and vectorized trainers)
- **Tools**: Replaced 26+ complex scripts with 3-4 simple utilities
- **Tests**: Reduced from 40+ to ~10 core functionality tests
- **Scripts**: Consolidated to 4-5 essential shell scripts

### Removed
- Phase completion summaries and reports
- Overnight training plans and logs
- Advanced BC features (ensemble, analysis, experiment tracking)
- Enhanced and vectorized PPO trainers
- Complex analysis and evaluation tools
- Phase-specific and feature-specific tests

### Kept
- Core environment, FSM controller, BC/PPO training pipelines
- Essential BC models (MLP, LSTM, GRU)
- Basic PPO trainer with standard features
- Essential test coverage for core functionality
- Simple training and evaluation scripts

## Previous Versions

### Phase 3 (Completed)
- Evaluation and analysis infrastructure
- Comprehensive test suite
- Advanced randomization
- Multi-condition testing

### Phase 2 (Completed)
- Training infrastructure
- BC and PPO training pipelines
- Model checkpoints and metrics
- TensorBoard integration

### Phase 1 (Completed)
- Data quality improvements
- Environment enhancements
- FSM controller implementation
- Basic BC training

