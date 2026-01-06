"""
ONE-COMMAND COMPREHENSIVE CV ANALYSIS

This script:
1. Regenerates test predictions if needed
2. Runs all comprehensive analyses
3. Generates all plots matching 20260102-163144

Usage:
    python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Path to CV results directory")
    parser.add_argument("--config", type=str, default="configs/multitask_example.yaml",
                        help="Path to config file used for training")
    args = parser.parse_args()
    
    cv_dir = Path(args.result_dir)
    
    print("="*80)
    print("COMPREHENSIVE 5-FOLD CV ANALYSIS")
    print("="*80)
    print(f"Directory: {cv_dir}\n")
    
    # Step 1: Check if predictions exist, regenerate if needed
    pred_file = cv_dir / "fold_1" / "test_predictions.npz"
    
    if not pred_file.exists():
        print("📊 Test predictions not found, regenerating...")
        print("-"*80)
        result = subprocess.run([
            sys.executable,
            "regenerate_cv_predictions.py",
            "--result-dir", str(cv_dir),
            "--config", args.config
        ])
        
        if result.returncode != 0:
            print("❌ Failed to regenerate predictions!")
            return 1
        
        print("\n✓ Predictions regenerated successfully!\n")
    else:
        print("✓ Test predictions already exist\n")
    
    # Step 2: Run comprehensive analysis
    print("="*80)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    result = subprocess.run([
        sys.executable,
        "analyze_cv_results_comprehensive.py",
        "--result-dir", str(cv_dir)
    ])
    
    if result.returncode != 0:
        print("❌ Analysis failed!")
        return 1
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)
    print(f"\nCheck {cv_dir} for all analysis plots:")
    print("  • prediction_scatter.png")
    print("  • error_analysis_by_time.png")
    print("  • error_vs_classification_confidence.png")
    print("  • valley_period_analysis.png")
    print("  • worst_predictions_report.txt")
    print("\nThese match the analysis from 20260102-163144!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
