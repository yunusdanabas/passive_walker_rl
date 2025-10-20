#!/usr/bin/env python3
"""
Unified Analysis Pipeline

Runs complete behavioral analysis and robustness testing.
Generates organized visual outputs in a single folder structure.

Usage:
    python analysis_code/run_analysis_pipeline.py \\
        --checkpoint checkpoints/model.pt \\
        --meta checkpoints/model_meta.json \\
        --episodes 10
"""

import argparse
from pathlib import Path
import json
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from behavioral_analysis import run_behavioral_analysis
from robustness_testing import run_robustness_testing


def create_output_structure(base_dir: Path):
    """Create organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"analysis_{timestamp}"
    
    # Create subdirectories
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    
    return run_dir


def create_summary_report(behavioral_results: dict, robustness_results: dict, output_dir: Path):
    """Create visual summary report."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Title and timestamp
    fig.suptitle('Complete Analysis Report', fontsize=18, fontweight='bold', y=0.98)
    
    # Behavioral metrics summary
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    beh_metrics = behavioral_results['metrics']
    reward_improvement = beh_metrics['reward_improvement']
    distance_diff = beh_metrics['nn_total_distance'] - beh_metrics['fsm_total_distance']
    
    summary_text = f"""
    BEHAVIORAL ANALYSIS SUMMARY
    
    NN Average Reward:     {beh_metrics['nn_avg_reward']:.3f}
    FSM Average Reward:    {beh_metrics['fsm_avg_reward']:.3f}
    Improvement:           {reward_improvement:+.3f} ({reward_improvement/abs(beh_metrics['fsm_avg_reward'])*100:+.1f}%)
    
    NN Total Distance:     {beh_metrics['nn_total_distance']:.2f}m
    FSM Total Distance:    {beh_metrics['fsm_total_distance']:.2f}m
    Difference:            {distance_diff:+.2f}m
    """
    
    ax1.text(0.05, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Robustness metrics summary
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    
    rob_metrics = robustness_results['metrics']
    
    robustness_text = f"""
    ROBUSTNESS TESTING SUMMARY
    
    Average NN Reward (across conditions):     {rob_metrics['avg_nn_reward']:.3f}
    Average FSM Reward (across conditions):    {rob_metrics['avg_fsm_reward']:.3f}
    Overall Improvement:                       {rob_metrics['improvement']:+.3f}
    
    Status: {'✅ NN outperforms FSM' if rob_metrics['improvement'] > 0 else '⚠️  FSM outperforms NN'}
    """
    
    ax2.text(0.05, 0.5, robustness_text, fontsize=12, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Key insights
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    
    insights = """
    KEY INSIGHTS
    
    • Neural network performance relative to FSM baseline
    • Control pattern analysis reveals behavioral differences
    • Robustness testing validates generalization capability
    • Visual outputs provide comprehensive understanding
    """
    
    ax3.text(0.1, 0.5, insights, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax3.set_title('Analysis Insights', fontsize=12, fontweight='bold', pad=10)
    
    # Generated files list
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    all_figures = behavioral_results['figures'] + robustness_results['figures']
    figures_text = "GENERATED FIGURES\n\n" + "\n".join([
        f"✓ {Path(f).name}" for f in all_figures
    ])
    
    ax4.text(0.1, 0.5, figures_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
    ax4.set_title('Output Files', fontsize=12, fontweight='bold', pad=10)
    
    # Save report
    output_path = output_dir / "figures" / "analysis_report.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Run unified analysis pipeline')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--meta', type=str, required=True,
                       help='Path to model metadata (.json file)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes per test (default: 10)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Base output directory (default: results)')
    parser.add_argument('--skip-behavioral', action='store_true',
                       help='Skip behavioral analysis')
    parser.add_argument('--skip-robustness', action='store_true',
                       help='Skip robustness testing')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("🚀 UNIFIED ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\n📦 Model:      {args.checkpoint}")
    print(f"📊 Episodes:   {args.episodes}")
    print(f"📁 Output:     {args.output_dir}")
    print()
    
    # Create output structure
    base_dir = Path(args.output_dir)
    run_dir = create_output_structure(base_dir)
    
    print(f"📂 Created output directory: {run_dir}")
    print()
    
    results = {
        'checkpoint': args.checkpoint,
        'meta': args.meta,
        'episodes': args.episodes,
        'timestamp': datetime.now().isoformat()
    }
    
    # Run behavioral analysis
    if not args.skip_behavioral:
        print("=" * 70)
        behavioral_results = run_behavioral_analysis(
            args.checkpoint,
            args.meta,
            run_dir / "figures",
            episodes=args.episodes
        )
        results['behavioral_analysis'] = behavioral_results
        print()
    else:
        print("⏭️  Skipping behavioral analysis")
        behavioral_results = None
    
    # Run robustness testing
    if not args.skip_robustness:
        print("=" * 70)
        robustness_results = run_robustness_testing(
            args.checkpoint,
            args.meta,
            run_dir / "figures",
            episodes=max(3, args.episodes // 2)  # Fewer episodes for robustness
        )
        results['robustness_testing'] = robustness_results
        print()
    else:
        print("⏭️  Skipping robustness testing")
        robustness_results = None
    
    # Create summary report
    if behavioral_results and robustness_results:
        print("=" * 70)
        print("📊 Creating summary report...")
        report_path = create_summary_report(behavioral_results, robustness_results, run_dir)
        results['summary_report'] = str(report_path)
        print(f"   ✅ Report saved: {report_path.name}")
        print()
    
    # Save metadata
    metadata_path = run_dir / "data" / "analysis_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {run_dir}")
    print(f"\n📊 Figures generated:")
    
    figure_count = 0
    if behavioral_results:
        for fig in behavioral_results['figures']:
            print(f"   • {Path(fig).name}")
            figure_count += 1
    if robustness_results:
        for fig in robustness_results['figures']:
            print(f"   • {Path(fig).name}")
            figure_count += 1
    if 'summary_report' in results:
        print(f"   • {Path(results['summary_report']).name}")
        figure_count += 1
    
    print(f"\n📈 Total figures: {figure_count}")
    print(f"💾 Metadata: {metadata_path}")
    print("\n" + "=" * 70)
    
    # Create symlink to latest
    latest_link = base_dir / "latest_analysis"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    
    print(f"🔗 Latest analysis: {latest_link}")
    print("=" * 70)


if __name__ == "__main__":
    main()

