#!/usr/bin/env python
import argparse
import os
from pathlib import Path

# Import your analysis class
from fast_jetclass.data.Analysis import ModelComparisonAnalysis

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to run on.")
    parser.add_argument("--nconst_values", nargs="+", type=int, default=[16, 32, 64],
                        help="List of nconst values to analyze.")
    parser.add_argument("--comparison", action="store_true", help="Compare multiple nconsts on the same plots.")
    args = parser.parse_args()

    # Point to the desired GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create our analysis object
    analyzer = ModelComparisonAnalysis(
        nconst_values=args.nconst_values,
        do_comparison=args.comparison,
        path_to_dir=args.data_root
    )

    # Run analysis for each nconst
    #for nconst in args.nconst_values:
        #analyzer.run_single_analysis(nconst)
        #print(f"Finished analysis for nconst={nconst}")
        #analyzer.test_pt_correlation(nconst)

    # Optionally compare models if do_comparison=True
    #analyzer.compare_models()
    analyzer.compare_efficiency_vs_rate( 
        fixed_pt=100, 
        fixed_eta=2.4,
        fixed_mass=0,
        pt_thresholds=None, 
        total_event_rate=30864, 
        decision_thresholds=None)
    analyzer.plot_output_dist_model(
        nconst = 8,
        pt_cut=0,
        fixed_eta=2.4,
        fixed_mass=0,
        )
    analyzer.plot_baseline_curve(
        nconst=8,
        fixed_eta=2.4,
        fixed_mass=0,
        pt_thresholds=None,
        total_event_rate=30864
    )
    analyzer.rate_vs_ptcuts(
        nconst=8,
        fixed_eta=2.4,
        fixed_mass=0,
        total_event_rate=30864,
        pt_thresholds=None,
    )
    analyzer.rate_vs_masscuts(
        nconst=8,
        fixed_eta=2.4,
        total_event_rate=30864,
        mass_thresholds=None,
    )
    #analyzer.compare_henry_with_mine(
    #    nconst=8,
    #)

if __name__ == "__main__":
    main()