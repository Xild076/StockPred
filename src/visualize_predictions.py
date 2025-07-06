import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.test_accuracy import AccuracyTester

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--save', action='store_true', help='Save plots to files instead of displaying')
    parser.add_argument('--output', type=str, default='prediction_analysis.png', 
                       help='Output filename for saved plots')
    
    args = parser.parse_args()
    
    try:
        print("Starting Model Prediction Visualization...")
        
        tester = AccuracyTester()
        print("Running accuracy testing...")
        results = tester.run_comprehensive_test()
        
        if args.save:
            print("Saving plots to files...")
            main_plot, dist_plot = tester.save_prediction_plots(results, args.output)
            print(f"Plots saved:")
            print(f"   Main plot: {main_plot}")
            print(f"   Distribution plot: {dist_plot}")
        else:
            print("Displaying interactive plots...")
            tester.plot_best_worst_predictions(results)
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
