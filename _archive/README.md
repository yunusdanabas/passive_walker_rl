# Archive Directory

This directory contains archived code, documentation, and tools that are kept for reference only.

## IMPORTANT

**Never import from `_archive/` in active code.** All code in this directory is historical and should not be used in production.

## Structure

- `_archive/legacy/` - Original legacy code from development
- `_archive/bc/` - Archived BC module features (advanced, experiment tracking, analysis)
- `_archive/ppo/` - Archived PPO trainers (enhanced, vectorized)
- `_archive/tools/` - Archived complex analysis and evaluation tools
- `_archive/tests/` - Archived phase, feature, and comprehensive tests
- `_archive/scripts/` - Archived specialized training and monitoring scripts

## Why Archive Instead of Delete?

Archiving code preserves:
- **Git history** - All commits remain in repository
- **Learning reference** - Can see how features evolved
- **Emergency rollback** - Can recover specific code if needed
- **Documentation** - Keeps context for why decisions were made

## Active Code

The active codebase has been simplified to focus on:
- Essential BC and PPO training pipelines
- Core environment and controllers
- Simple utility scripts and tools
- Minimal test suite covering core functionality

