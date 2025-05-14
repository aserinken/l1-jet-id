import os
import numpy as np
import awkward as ak
import sklearn.model_selection
from tqdm import tqdm
from pathlib import Path
import uproot
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf

class ModelComparisonAnalysis:

    """
    analysis class for comparing multiple models
    with different parameters (like 'nconst' values).
    """

    def __init__(
        self, 
        path_to_dir: str,
        nconst_values, 
        do_comparison=True,
    ):
        super().__init__()
        self.nconst_values = nconst_values
        self.do_comparison = do_comparison
        self.model_path = Path(path_to_dir) 
        self.dir_path = self.model_path / "data" / "test"
        
    def get_paths_data(self, nconst_list=None, dir_path=None):
        """
        Returns a dictionary keyed by each nconst, with subkeys for:
            '3d_data', '3d_labels', '2d_data', '2d_labels'.
        Example structure:
        {
        16: {
            '3d_data': Path(.../proc_test_16const.npy),
            '3d_labels': Path(.../proc_labels_test_16const.npy),
            '2d_data': Path(.../proc_top_test_16const.npy),
            '2d_labels': Path(.../proc_top_labels_test_16const.npy)
        },
        32: { ... }, ...
        }
        """
        if nconst_list is None:
            nconst_list = self.nconst_values  # fallback to class-level list

        if dir_path is None:
            dir_path = self.dir_path

        if not dir_path.exists():
            raise FileNotFoundError(f"Data path {dir_path} does not exist.")

        all_paths = {}

        for n in nconst_list:
            # path to processed folder for each nconst
            path = dir_path / f"{n}const" / "processed"
            if not path.exists():
                raise FileNotFoundError(f"Data path {path} does not exist.")

            # gather relevant test files
            files = [
                f for f in os.listdir(path) 
                if "test" in f and not f.startswith("._")
            ]
            if not files:
                print(f"Warning: No test files found under {path}")
                # Initialize empty entry and continue
                all_paths[n] = {
                    "3d_data": None,
                    "3d_labels": None,
                    "2d_data": None,
                    "2d_labels": None
                }
                continue

            # Initialize structure
            entry = {
                "3d_data": None,
                "3d_labels": None,
                "2d_data": None,
                "2d_labels": None
            }

            # Assign files into each bucket (3D vs 2D, data vs labels)
            for fn in files:
                full_path = path / fn

                # Check if "top" => 2D data, else 3D data
                if "top" in fn:
                    if "labels" in fn:
                        entry["2d_labels"] = full_path
                    else:
                        entry["2d_data"] = full_path
                else:
                    if "labels" in fn:
                        entry["3d_labels"] = full_path
                    else:
                        entry["3d_data"] = full_path

            all_paths[n] = entry

        return all_paths


    def load_data(self, nconst, all_paths=None):
        """
        Load the 3D and 2D data (and labels) for a specific nconst.
        Returns a dictionary with keys:
        '3d_data', '3d_labels', '2d_data', '2d_labels'.
        """
        if all_paths is None:
            all_paths = self.get_paths_data()

        if nconst not in all_paths:
            raise ValueError(f"No data paths found for nconst={nconst}")

        paths = all_paths[nconst]

        data_dict = {
            "3d_data": None,
            "3d_labels": None,
            "2d_data": None,
            "2d_labels": None
        }

        #Load each file if it exists
        if paths["3d_data"] is not None and paths["3d_data"].is_file():
            data_dict["3d_data"] = np.load(paths["3d_data"], allow_pickle=True)
        if paths["3d_labels"] is not None and paths["3d_labels"].is_file():
            data_dict["3d_labels"] = np.load(paths["3d_labels"], allow_pickle=True)

        if paths["2d_data"] is not None and paths["2d_data"].is_file():
            data_dict["2d_data"] = np.load(paths["2d_data"], allow_pickle=True)
        if paths["2d_labels"] is not None and paths["2d_labels"].is_file():
            data_dict["2d_labels"] = np.load(paths["2d_labels"], allow_pickle=True)

        return data_dict
    
    def get_paths_models(self, nconst_list=None, model_root=None):
        """
        Returns a dictionary keyed by each nconst,
        giving the path to its model directory or file.

        Example structure:
        {
            16: Path(.../SC8SCAN_deepsets_8bit_16const/kfolding1/model/model.h5),
            32: Path(.../SC8SCAN_deepsets_8bit_32const/kfolding1/model/model.h5),
            ...
        }
        If no file is found, the value is None.
        """
        if nconst_list is None:
            nconst_list = self.nconst_values

        # Base directory where your models live
        if model_root is None:
            model_root = self.model_path / "scripts" / "trained_deepsets"

        all_models = {}
        for n in nconst_list:
            model_dir = model_root / f"SC8SCAN_deepsets_8bit_{n}const" / "kfolding1"
            if not model_dir.is_dir():
                print(f"Warning: Model directory not found for nconst={n}: {model_dir}")
                all_models[n] = None
                continue

            # Check if saved_model.pb exists in this directory
            pb_file = model_dir / "saved_model.pb"
            if not pb_file.is_file():
                print(f"Warning: No saved_model.pb file found in {model_dir}")
                all_models[n] = None
            else:
                all_models[n] = model_dir

        return all_models
    
    def load_model_for_nconst(self, nconst, all_models=None):
        """
        Load the model file for a specific nconst from all_models.
        Returns a loaded model object or None if not found.
        """
        if all_models is None:
            all_models = self.get_paths_models()

        model_path = all_models.get(nconst)
        if model_path is None:
            print(f"No model path for nconst={nconst}.")
            return None

        from tensorflow.keras.models import load_model
        print(f"Loading model from {model_path}")
        model = load_model(str(model_path), compile=False)
        return model
        

    def run_single_analysis(self, nconst):
        """
        Example: load data, load model, evaluate, store metrics.
        """
        # 1) Load data
        data = self.load_data(nconst)  
        x_test_3d = data["3d_data"]
        y_test_3d = data["3d_labels"]

        if x_test_3d is None or y_test_3d is None:
            print(f"No 3D test data/labels for nconst={nconst}. Skipping analysis.")
            self.results[nconst] = {"accuracy": None, "auroc": None}
            return

        # 2) Load model
        all_models = self.get_paths_models()
        model = self.load_model_for_nconst(nconst, all_models)
        if model is None:
            print(f"No model loaded for nconst={nconst}. Skipping analysis.")
            self.results[nconst] = {"accuracy": None, "auroc": None}
            return

        # 3) Generate predictions with the actual model
        # Suppose the model outputs shape (N, 2): background vs signal probabilities
        y_pred_probs = model.predict(x_test_3d)  # shape = (numSamples, 2)
        # Probability of “signal” is column index 1
        signal_scores = y_pred_probs[:, 1]

        # y_test_3d is one-hot (e.g. [[1,0],[0,1],...]),
        # convert it to integer labels [0 or 1]
        y_true = np.argmax(y_test_3d, axis=1)

        # binary predictions from signal_scores
        y_pred = (signal_scores > 0.5).astype(int)


        acc = accuracy_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, signal_scores)
        except ValueError:
            auroc = None  


        metrics = {"accuracy": acc, "auroc": auroc}
        self.results[nconst] = metrics
        print(f"[nconst={nconst}] accuracy={acc}, auroc={auroc}")

    def compare_models(self):
        """
        Compare ROC curves for each 'nconst' model on a single plot.
        The plot is saved in a 'plots/roc_comparison' directory under the model path.
        """
        import os
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        # Create an output directory for plots
        outdir = self.model_path / "plots" / "roc_comparison"
        os.makedirs(outdir, exist_ok=True)

        plt.figure(figsize=(8, 6))

        for nconst in self.nconst_values:
            # Load test data for this nconst
            data = self.load_data(nconst)
            x_test_3d = data["3d_data"]
            y_test_3d = data["3d_labels"]

            if x_test_3d is None or y_test_3d is None:
                print(f"Skipping nconst={nconst}: missing test data/labels.")
                continue

            # Convert one-hot labels to integer labels
            y_true = np.argmax(y_test_3d, axis=1)

            # Load the corresponding model
            all_models = self.get_paths_models()
            model = self.load_model_for_nconst(nconst, all_models)
            if model is None:
                print(f"Skipping nconst={nconst}: no model loaded.")
                continue

            # Generate predictions
            y_pred_probs = model.predict(x_test_3d)  
            signal_scores = y_pred_probs[:, 1]

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, signal_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"nconst={nconst} (AUC={roc_auc:.4f})")

        # Plot chance line
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")

        # Save the plot
        outpath = os.path.join(outdir, "roc_comparison.png")
        plt.savefig(outpath)
        plt.close()
        
        print(f"ROC curves saved to {outpath}")

    def test_pt_correlation(self, nconst=None):
        """
        Test the correlation between the 3D sum pt and the top-level (2D) pt.
        """
        print("=== Starting test_pt_correlation ===")
        data = self.load_data(nconst)

        if nconst is None:
            nconst = self.nconst_values[0]
        data = self.load_data(nconst)
        
        if data["3d_data"] is None or data["2d_data"] is None:
            print(f"Missing 3D or 2D data for nconst={nconst}.")
            return
        else:
            print(f"3D data shape: {data['3d_data'].shape}")
            print(f"2D data shape: {data['2d_data'].shape}")
        
        threeD = data["3d_data"]
        twoD = data["2d_data"]

        n_events = min(5, threeD.shape[0])
        for i in range(n_events):
            sum_pt_3d = np.sum(threeD[i][:, 0])
            top_pt = twoD[i][0] if isinstance(twoD[i], (list, np.ndarray)) else twoD[i]
            print(f"Event {i}: sum(pt, 3d) = {sum_pt_3d:.2f}, top-level pt = {top_pt:.2f}")


    def plot_efficiency_vs_rate(self, nconst, fixed_pt=0, fixed_eta = 2.4, pt_thresholds=None, 
                                total_event_rate=30864, decision_thresholds=None):
        """
        For a given nconst, compute two curves:
        
        1. Baseline (No Tagger): 
        Scan over a range of top-level pt cuts (from the 2D data) and, for each cut,
        compute the signal efficiency and corresponding background rate.
        
        2. With Tagger:
        First, apply a fixed top-level pt cut (fixed_pt) to the events.
        Then, scan across a range of decision thresholds (for the model's signal probability)
        and compute the signal efficiency and rate.
        
        The method then plots both curves.
        
        Arguments:
        nconst: Constituent value (e.g., 16, 32, etc.).
        fixed_pt: A fixed top-level pt cut to apply for the tagger case.
        pt_thresholds: Array of top-level pt thresholds (units of your 2D data) used for baseline.
                        If None, a default range is used.
        total_event_rate: Overall event rate (in kHz) to scale the background rate.
        decision_thresholds: Array of decision thresholds to scan for the tagger.
                            If None, a default range from 0 to 1 is used.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Default range for baseline pt thresholds.
        if pt_thresholds is None:
            pt_thresholds = np.arange(0, 1000, 10)  # adjust limits and step as needed

        # Default range for decision thresholds.
        if decision_thresholds is None:
            decision_thresholds = np.linspace(0, 1, 1000)

        # Load data for the given nconst.
        data = self.load_data(nconst)
        if data["3d_data"] is None or data["2d_data"] is None or data["3d_labels"] is None:
            print(f"Missing some data for nconst={nconst}.")
            return

        # 2D top-level data and 3D labels.
        twoD = data["2d_data"]
        # Convert 3D one-hot labels into binary ones.
        y_true = np.argmax(data["3d_labels"], axis=1)
        x3d = data["3d_data"]

        total_signal = np.sum(y_true == 1)
        total_background = np.sum(y_true == 0)
        
        # --- Baseline: Scan over pt thresholds applied on 2D data ---
        baseline_eff = []
        baseline_rate = []

        for thresh in pt_thresholds:
            mask = (twoD[:, 0] >= thresh) & (np.abs(twoD[:, 1]) < fixed_eta)
            sig_count = np.sum((y_true == 1) & mask)
            bkg_count = np.sum((y_true == 0) & mask)
            eff = sig_count / (total_signal)
            rate = (bkg_count / (total_background)) * total_event_rate
            baseline_eff.append(eff)
            baseline_rate.append(rate)

        # --- With Tagger: First, apply a fixed 2D pt cut, then scan decision thresholds ---
        fixed_mask = (twoD[:, 0] >= fixed_pt) & (np.abs(twoD[:, 1]) < fixed_eta)
        num_fixed_events = np.sum(fixed_mask)
        print(f"Number of events passing fixed top-level pt cut of {fixed_pt}: {num_fixed_events}")
        if num_fixed_events == 0:
            print(f"No events pass the fixed top-level pt cut of {fixed_pt}.")
            return
        # Get events passing the fixed 2D pt cut.
        y_true_fixed = y_true[fixed_mask]
        # Obtain model predictions on all 3D data and then select the fixed events.
        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"No model loaded for nconst={nconst}. Cannot compute tagger performance.")
            return
        # Compute raw model predictions (logits)
        y_pred_probs = model.predict(x3d)
        # Apply softmax to get probabilities (note: y_pred_probs shape = (numSamples, numClasses))
        probs = tf.nn.softmax(y_pred_probs).numpy()
        # Extract probability of the "signal" class (assumed at column index 1)
        signal_scores = probs[:, 1]
        # Select the fixed events from the computed probabilities
        signal_scores_fixed = signal_scores[fixed_mask]

        print("Signal score stats for fixed events:")
        print("  Min:", np.min(signal_scores_fixed))
        print("  Max:", np.max(signal_scores_fixed))
        print("  Mean:", np.mean(signal_scores_fixed))
        print("  Std:", np.std(signal_scores_fixed))

        tagger_eff = []
        tagger_rate = []
        for dt in decision_thresholds:
            # Apply decision threshold on the model's signal scores.
            pred_fixed = (signal_scores_fixed > dt).astype(int)
            # Compute efficiency and rate on the fixed set.
            eff = np.sum((pred_fixed == 1) & (y_true_fixed == 1)) / (np.sum(y_true_fixed == 1))
            rate = (np.sum((pred_fixed == 1) & (y_true_fixed == 0)) / (np.sum(y_true_fixed == 0))) * total_event_rate
            tagger_eff.append(eff)
            tagger_rate.append(rate)

        # Plotting the results.
        plt.figure(figsize=(10, 6))
        # Plot baseline: efficiency vs. rate as pt threshold varies.
        plt.semilogy(baseline_rate, baseline_eff, label="Baseline (varying total pt cut)",
                    marker="o", linestyle="dashed", color="blue", markersize=5)
        # Plot with tagger: efficiency vs. rate as decision threshold varies.
        plt.semilogy(tagger_rate, tagger_eff, label=f"With Tagger (fixed total pt cut = {fixed_pt})",
                    marker="s", linestyle="solid", color="red", markersize=5)
        plt.xlabel("Trigger Rate [kHz]")
        plt.xlim(0,100)
        plt.ylabel("Signal Efficiency")
        plt.title(f"Signal Efficiency vs Trigger Rate (nconst={nconst})")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", linewidth=0.5)
        
        # Find the index in pt_thresholds closest to fixed_pt
        fixed_index = np.abs(pt_thresholds - fixed_pt).argmin()
        baseline_rate_fixed = baseline_rate[fixed_index]
        baseline_eff_fixed = baseline_eff[fixed_index]

        # Add a special marker for this point on the baseline curve
        plt.scatter(baseline_rate_fixed, baseline_eff_fixed, color='green', marker='*', s=200,
                    label=f"Baseline at fixed_pt = {fixed_pt}")
        plt.annotate(f"fixed_pt: {fixed_pt}", 
                    (baseline_rate_fixed, baseline_eff_fixed),
                    textcoords="offset points", 
                    xytext=(10, -10),
                    ha="left", color='green')

        # Save the figure.
        save_dir = os.path.join(self.model_path, "plots", "efficiency_vs_rate")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"efficiency_vs_rate_{nconst}const.png")
        plt.savefig(save_path)
        print(f"Efficiency vs Rate plot saved to {save_path}")
        plt.show()

    def compare_efficiency_vs_rate(self, fixed_pt=100, fixed_eta=2.4, fixed_mass=0, pt_thresholds=None, 
                                total_event_rate=30864, decision_thresholds=None):
        """
        Generate a single plot comparing the tagger efficiency vs trigger rate curves across different
        nconst models – the baseline curve (computed via top-level pt cuts on the 2D data) is plotted only once.
        
        For the baseline, we scan over a range of top-level pt cuts (with |eta| < fixed_eta)
        and compute the signal efficiency and background rate. The baseline is expected to be identical
        for all nconst values.
        
        For the tagger curve, a fixed top-level pt cut (fixed_pt) is applied and we scan over decision thresholds
        on the model's signal probability.
        
        Arguments:
        fixed_pt: Fixed top-level pt cut to use in the tagger part.
        fixed_eta: Maximum |eta| allowed (applied to both baseline and fixed selections).
        pt_thresholds: Array of pt thresholds for baseline; default is np.arange(10, 1000, 10).
        total_event_rate: Scale factor for background rate (in kHz).
        decision_thresholds: Array of decision thresholds; default is np.linspace(0,1,1000).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import tensorflow as tf

        if pt_thresholds is None:
            pt_thresholds = np.arange(0, 1000, 4)

        if decision_thresholds is None:
            decision_thresholds = np.linspace(0, 1, 300)

        plt.figure(figsize=(12, 8))

        # Compute the baseline curve only once using the data from the first nconst value.
        first_nconst = self.nconst_values[0]
        base_data = self.load_data(first_nconst)
        if base_data["3d_labels"] is None or base_data["2d_data"] is None:
            print(f"Missing data for nconst={first_nconst}. Cannot compute baseline.")
            return
        twoD_base = base_data["2d_data"]
        y_true_base = np.argmax(base_data["3d_labels"], axis=1)
        total_signal_base = np.sum(y_true_base == 1)
        total_background_base = np.sum(y_true_base == 0)
        baseline_eff = []
        baseline_rate = []
        for thresh in pt_thresholds:
            mask = (twoD_base[:, 0] >= thresh) #& (np.abs(twoD_base[:, 1]) < fixed_eta) & (twoD_base[:, 3] >= fixed_mass) 
            sig_count = np.sum((y_true_base == 1) & mask)
            bkg_count = np.sum((y_true_base == 0) & mask)
            eff = sig_count / (total_signal_base)
            rate = (bkg_count / (total_background_base)) * total_event_rate
            baseline_eff.append(eff)
            baseline_rate.append(rate)

        # Find the index corresponding to the pt cut of 126.
        pt_cut_value = 126
        idx_126 = np.abs(pt_thresholds - pt_cut_value).argmin()
        baseline_rate_126 = baseline_rate[idx_126]
        baseline_eff_126 = baseline_eff[idx_126]
        print(f"Baseline at pt>{pt_cut_value} GeV: Rate = {baseline_rate_126:.2f} kHz, Efficiency = {baseline_eff_126:.3f}")

        # Optionally, annotate this point on the plot:
        plt.scatter(baseline_rate_126, baseline_eff_126, color='purple', marker="*", s=200,
                    zorder=12, label=f"Baseline at pt>{pt_cut_value} GeV")
        plt.annotate(f"pt>{pt_cut_value}: {baseline_rate_126:.2f} kHz",
                    (baseline_rate_126, baseline_eff_126),
                    textcoords="offset points",
                    xytext=(10, -10),
                    ha="left", color='purple')
        # Plot the baseline curve.
        plt.plot(baseline_rate, baseline_eff, label="Baseline (varrying total pt cut)",
                    marker="o", linestyle="dashed", color="blue", markersize=4)

        # Mark the baseline point corresponding to fixed_pt.
        fixed_index = np.abs(pt_thresholds - fixed_pt).argmin()
        plt.scatter(baseline_rate[fixed_index], baseline_eff[fixed_index],
                    color="green", marker="*", s=fixed_pt, zorder = 10,
                    label=f"Baseline at fixed_pt = {fixed_pt}")

        # Now, for each nconst value, compute and plot the tagger curve.
        for nconst in self.nconst_values:
            data = self.load_data(nconst)
            if data["3d_data"] is None or data["2d_data"] is None or data["3d_labels"] is None:
                print(f"Missing data for nconst={nconst}. Skipping model.")
                continue

            twoD = data["2d_data"]
            y_true = np.argmax(data["3d_labels"], axis=1)
            x3d = data["3d_data"]

            # --- With Tagger: Apply fixed pt & |eta| condition ---
            fixed_mask = (twoD[:, 0] >= fixed_pt) #& (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass) 
            num_fixed_events = np.sum(fixed_mask)
            print(f"[nconst={nconst}] Fixed pt cut {fixed_pt} yields {num_fixed_events} events.")
            if num_fixed_events == 0:
                print(f"No events pass the fixed top-level pt cut {fixed_pt} for nconst={nconst}.")
                continue

            y_true_fixed = y_true[fixed_mask]

            model = self.load_model_for_nconst(nconst)
            if model is None:
                print(f"No model loaded for nconst={nconst}. Skipping tagger part.")
                continue

            y_pred_probs = model.predict(x3d)
            # Apply softmax to get probabilities.
            probs = tf.nn.softmax(y_pred_probs).numpy()
            signal_scores = probs[:, 1]
            signal_scores_fixed = signal_scores[fixed_mask]

            print(f"[nconst={nconst}] Signal score stats (fixed events): "
                f"Min={np.min(signal_scores_fixed):.3f}, Mean={np.mean(signal_scores_fixed):.3f}, Max={np.max(signal_scores_fixed):.3f}")

            tagger_eff = []
            tagger_rate = []
            for dt in decision_thresholds:
                pred_fixed = (signal_scores_fixed > dt).astype(int)
                eff = np.sum((pred_fixed == 1) & (y_true_fixed == 1)) / total_signal_base
                #eff = np.sum((pred_fixed == 1) & (y_true_fixed == 1)) / (np.sum(y_true_fixed == 1))
                rate = (np.sum((pred_fixed == 1) & (y_true_fixed == 0)) / total_background_base) * total_event_rate
                #rate = (np.sum((pred_fixed == 1) & (y_true_fixed == 0)) / (np.sum(y_true_fixed == 0))) * total_event_rate
                tagger_eff.append(eff)
                tagger_rate.append(rate)

            # Plot the tagger curve for this nconst.
            plt.plot(tagger_rate, tagger_eff, linestyle="-",
                        marker="s", markersize=4, label=f"With Tagger {nconst}const")

        plt.xlabel("Trigger Rate [kHz]")
        plt.ylabel("Signal Efficiency")
        plt.xlim(0, 120)
        plt.title("Signal Efficiency vs Trigger Rate Comparison Across nconst Values")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, linestyle="--", linewidth=0.5)

        # Save the final comparison plot.
        save_dir = os.path.join(self.model_path, "plots", "efficiency_vs_rate_comparison")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "comparison.png")
        plt.savefig(save_path)
        print(f"Efficiency vs Rate comparison plot saved to {save_path}")
        plt.show()

    def compare_ptcuts_for_one_model(
        self,
        nconst = 16,
        pt_cuts=(80, 100, 150),      # 3 hardware pT cuts you want to study
        fixed_eta=2.4,
        fixed_mass=10,               # mass cut on the 2D data
        pt_thresholds=None,          # for the baseline scan
        total_event_rate=30864,
        decision_thresholds=None):
        """
        For a single nconst model:
            • draw the common *baseline* curve  (vary only the top‑level pT cut)
            • overlay three tagger curves, each built with a *different* fixed pT cut
        That lets you visualise what tightening / loosening the hardware cut
        does to the 'NN vs plain‑pT' trade‑off.

        Args
        ----
        nconst               : which model to load (must exist in self.nconst_values)
        pt_cuts (iterable)   : the fixed pT thresholds you want to test (≤3 recommended)
        fixed_eta            : |η| window applied everywhere
        pt_thresholds        : np.arange(10, 1_000, 10) by default
        total_event_rate     : background normalisation (kHz)
        decision_thresholds  : np.linspace(0, 1, 1_000) by default
        """
        import numpy as np, matplotlib.pyplot as plt, tensorflow as tf, os

        if pt_thresholds is None:
            pt_thresholds = np.arange(0, 1000, 10)
        if decision_thresholds is None:
            decision_thresholds = np.linspace(0, 1, 300)

        # -------------- load data & model once ---------------------------------
        assert nconst in self.nconst_values, f"nconst={nconst} not trained."
        data = self.load_data(nconst)
        if any(data[k] is None for k in ("3d_data", "3d_labels", "2d_data")):
            print("Missing data - aborting.")
            return

        twoD   = data["2d_data"]
        y_true = np.argmax(data["3d_labels"], axis=1)
        x3d    = data["3d_data"]

        total_signal_base     = np.sum(y_true == 1)
        total_background_base = np.sum(y_true == 0)

        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"Model for nconst={nconst} not found.")
            return
        probs = tf.nn.softmax(model.predict(x3d), axis=1).numpy()
        signal_scores = probs[:, 1]

        # -------------- baseline curve (varied pT cut) -------------------------
        baseline_eff, baseline_rate = [], []
        for thresh in pt_thresholds:
            mask      = (twoD[:, 0] >= thresh) & (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass) 
            sig_count = np.sum((y_true == 1) & mask)
            bkg_count = np.sum((y_true == 0) & mask)
            baseline_eff .append(sig_count / total_signal_base)
            baseline_rate.append((bkg_count / total_background_base) * total_event_rate)

        plt.figure(figsize=(12, 8))
        plt.plot(baseline_rate, baseline_eff,
                    "o--", color="tab:blue", label="Baseline (vary pT)")

        # -------------- loop over the chosen hardware cuts ---------------------
        colours = ["tab:green", "tab:orange", "tab:red"]
        for cut, c in zip(pt_cuts, colours):
            mask_hw      = (twoD[:, 0] >= cut) & (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass) 
            y_true_hw    = y_true[mask_hw]
            sig_scores_hw = signal_scores[mask_hw]

            # mark the star on the baseline for this cut
            idx_closest = np.abs(pt_thresholds - cut).argmin()
            plt.scatter(baseline_rate[idx_closest], baseline_eff[idx_closest],
                        marker="*", s=cut, color=c, zorder=11,
                        label=f"Baseline star pT>{cut} GeV")

            # tagger curve for this pT cut
            tag_eff, tag_rate = [], []
            for dt in decision_thresholds:
                pred = (sig_scores_hw > dt)
                sig_keep = np.sum(pred & (y_true_hw == 1))
                bkg_keep = np.sum(pred & (y_true_hw == 0))

                # *global* denominators to match the baseline
                tag_eff .append(sig_keep / total_signal_base)
                tag_rate.append((bkg_keep / total_background_base) * total_event_rate)

            plt.plot(tag_rate, tag_eff, "-", color=c,
                        label=f"With tagger pT>{cut} GeV")

        # -------------- cosmetics & output -------------------------------------
        plt.xlabel("Trigger Rate [kHz]")
        plt.ylabel("Signal Efficiency")
        plt.xlim(0, 300)
        plt.title(f"nconst = {nconst}: Tagger vs baseline for different pT cuts")
        plt.grid(True, ls="--", lw=0.5)
        plt.legend(fontsize=9, loc="best")

        out_dir  = os.path.join(self.model_path, "plots", f"ptcut_scan_n{nconst}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ptcut_comparison.png")
        plt.savefig(out_path);  print("Saved →", out_path)
        plt.show()


    def plot_output_dist_model(
        self,
        nconst = 16,
        pt_cut=0,
        fixed_eta=2.4,
        fixed_mass = 20
        ):
        """
        Plot the output distribution of the model for a given nconst.
        
        Args:
            nconst: Constituent value (e.g., 16, 32, etc.).
            pt_cut: Fixed top-level pt cut to apply.
            fixed_eta: Maximum |eta| allowed.
            pt_thresholds: Array of pt thresholds for baseline; default is np.arange(10, 1000, 10).
            total_event_rate: Scale factor for background rate (in kHz).
            decision_thresholds: Array of decision thresholds; default is np.linspace(0,1,1000).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import tensorflow as tf



        # Load data for the given nconst.
        data = self.load_data(nconst)
        if data["3d_data"] is None or data["2d_data"] is None or data["3d_labels"] is None:
            print(f"Missing some data for nconst={nconst}.")
            return

        twoD = data["2d_data"]
        x3d = data["3d_data"]
        print(f"2D data mass unique: {np.unique(twoD[:, 3])}")

        # Apply the fixed top-level pT cut and |eta| condition.
        fixed_mask = (twoD[:, 0] >= pt_cut) & (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass) 
        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"No model loaded for nconst={nconst}. Cannot compute tagger performance.")
            return
        probs = tf.nn.softmax(model.predict(x3d), axis=1).numpy()
        signal_scores = probs[:, 1]
        signal_scores_fixed = signal_scores[fixed_mask]

        # Plot the distribution of signal scores for the fixed events.
        plt.figure(figsize=(10, 6))
        plt.hist(signal_scores_fixed, bins=50, density=True, alpha=0.7, color='blue', label='Signal Scores')
        #plt.hist(twoD[:,3], bins=50, density=True, alpha=0.5, color='red', label='Mass Distribution')
        plt.xlabel("Signal Score")
        plt.ylabel("Density")
        plt.title(f"Signal Score Distribution for nconst={nconst} with pT cut > {pt_cut} GeV")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        # Save the figure.
        save_dir = os.path.join(self.model_path, "plots", "output_distribution")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"output_distribution_{nconst}.png")
        plt.savefig(save_path)
        print(f"Output distribution plot saved to {save_path}")
        plt.show()

    def plot_baseline_curve(
            self,
            nconst=8,
            fixed_eta=2.4,
            fixed_mass=20,
            pt_thresholds=None,
            total_event_rate=30864
        ):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import tensorflow as tf

        if pt_thresholds is None:
            pt_thresholds = np.arange(0, 1000, 5)

        plt.figure(figsize=(12, 8))

        base_data = self.load_data(nconst)
        if base_data["3d_labels"] is None or base_data["2d_data"] is None:
            print(f"Missing data for nconst={nconst}. Cannot compute baseline.")
            return
        twoD_base = base_data["2d_data"]
        y_true_base = np.argmax(base_data["3d_labels"], axis=1)
        total_signal_base = np.sum(y_true_base == 1)
        total_background_base = np.sum(y_true_base == 0)

        baseline_eff = []
        baseline_rate = []
        for thresh in pt_thresholds:
            mask = (twoD_base[:, 0] >= thresh) #& (np.abs(twoD_base[:, 1]) < fixed_eta) & (twoD_base[:, 3] >= fixed_mass) 
            sig_count = np.sum((y_true_base == 1) & mask)
            bkg_count = np.sum((y_true_base == 0) & mask)
            eff = sig_count / (total_signal_base) 
            rate = (bkg_count / (total_background_base)) * total_event_rate
            baseline_eff.append(eff)
            baseline_rate.append(rate)

        # Find the index corresponding to the pt cut of 126.
        pt_cut_value = 126
        idx_126 = np.abs(pt_thresholds - pt_cut_value).argmin()
        baseline_rate_126 = baseline_rate[idx_126]
        baseline_eff_126 = baseline_eff[idx_126]
        print(f"Baseline at pt>{pt_cut_value} GeV: Rate = {baseline_rate_126:.2f} kHz, Efficiency = {baseline_eff_126:.3f}")

        # Mark 126 point on the baseline curve
        plt.scatter(baseline_rate_126, baseline_eff_126, color='purple', marker="*", s=200,
                    zorder=12, label=f"Baseline at pt>{pt_cut_value} GeV")
        plt.annotate(f"pt>{pt_cut_value}: {baseline_rate_126:.2f} kHz",
                    (baseline_rate_126, baseline_eff_126),
                    textcoords="offset points",
                    xytext=(10, -10),
                    ha="left", color='purple')
        # Plot the baseline curve.
        plt.plot(baseline_rate, baseline_eff, label="Baseline (varrying total pt cut)",
                    marker="o", linestyle="dashed", color="blue", markersize=4)

        plt.xlabel("Trigger Rate [kHz]")
        plt.ylabel("Signal Efficiency")
        plt.xlim(0, 300)
        plt.title(f"nconst = {nconst}: Tagger vs baseline for different pT cuts")
        plt.grid(True, ls="--", lw=0.5)
        plt.legend(fontsize=9, loc="best")

        out_dir  = os.path.join(self.model_path, "plots", f"baseline_n{nconst}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "baseline_plot.png")
        plt.savefig(out_path);  print("Saved →", out_path)
        plt.show()
    
    
    
    """""
    def plot_rate_vs_pt_for_tagger_operating_points(self, nconst, pt_cuts_to_scan,  # e.g. np.arange(80, 151, 10)
                                                fixed_eta=2.4, fixed_mass=0, 
                                                decision_thresholds=None, 
                                                total_event_rate=32000, 
                                                target_efficiencies=(0.80, 0.50, 0.20)):
        ""
        For a given nconst model, vary the hardware (top-level) pT cut and, for each, compute the tagger curve 
        (by scanning decision thresholds). Then extract the trigger rate that corresponds to target signal efficiencies
        (e.g. loose/medium/tight = 80/50/20%) and plot rate vs. hardware pT cut for these operating points.
        
        Args:
        nconst             : The constituent value/model to analyze.
        pt_cuts_to_scan    : Array-like of fixed pT cut values (hardware cut) to test.
        fixed_eta          : Absolute eta window.
        fixed_mass         : Mass cut requirement.
        decision_thresholds: Array of decision threshold values; if None, defaults are used.
        total_event_rate   : Scale factor for background rate (kHz).
        target_efficiencies: Tuple of target signal efficiencies (e.g. (0.80, 0.50, 0.20)).
        ""
        import numpy as np, matplotlib.pyplot as plt, tensorflow as tf, os

        if decision_thresholds is None:
            decision_thresholds = np.linspace(0, 1, 300)

        # Load data once for this model.
        data = self.load_data(nconst)
        if any(data[k] is None for k in ("3d_data", "3d_labels", "2d_data")):
            print(f"Missing data for nconst={nconst}.")
            return

        twoD   = data["2d_data"]
        y_true = np.argmax(data["3d_labels"], axis=1)
        x3d    = data["3d_data"]
        total_signal = np.sum(y_true == 1)
        total_background = np.sum(y_true == 0)

        # Load the model.
        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"No model loaded for nconst={nconst}.")
            return
        # Get tagger probabilities from 3D data.
        y_pred_probs = model.predict(x3d)
        probs = tf.nn.softmax(y_pred_probs, axis=1).numpy()
        signal_scores = probs[:, 1]

        # Initialize dict to store rates for each target efficiency.
        results = {teff: [] for teff in target_efficiencies}
        pt_cut_list = []

        # Loop over the hardware pT cut values.
        for hard_pt in pt_cuts_to_scan:
            # Apply hardware cut.
            mask = (twoD[:, 0] >= hard_pt) & (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass)
            num_events = np.sum(mask)
            if num_events == 0:
                continue
            
            pt_cut_list.append(hard_pt)
            # For these events, get the tagger signal scores and true labels.
            y_true_fixed = y_true[mask]
            tagger_scores = signal_scores[mask]

            # Scan over decision thresholds to build tagger performance curve.
            tagger_eff = []
            tagger_rate = []
            for dt in decision_thresholds:
                pred = (tagger_scores > dt).astype(int)
                eff = np.sum((pred == 1) & (y_true_fixed == 1)) / (np.sum(y_true_fixed == 1))
                rate = (np.sum((pred == 1) & (y_true_fixed == 0)) / (np.sum(y_true_fixed == 0))) * total_event_rate
                tagger_eff.append(eff)
                tagger_rate.append(rate)
            tagger_eff = np.array(tagger_eff)
            tagger_rate = np.array(tagger_rate)
            
            # For each target efficiency, find the tagger rate closest to that efficiency.
            for teff in target_efficiencies:
                idx = np.abs(tagger_eff - teff).argmin()
                results[teff].append(tagger_rate[idx])
        
        # Now, plot the trigger rate vs hardware pT cut for each target efficiency.
        plt.figure(figsize=(10, 6))
        for teff, rates in results.items():
            plt.plot(pt_cut_list, rates, marker='o', linestyle='-', label=f"Tagger, target eff = {teff*100:.0f}%")
        
        plt.xlabel("Hardware pT cut (GeV)")
        plt.ylabel("Trigger Rate (kHz)")
        plt.title(f"Rate vs pT for Tagger Operating Points (nconst={nconst})")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", linewidth=0.5)
        
        # Save and show the plot.
        save_dir = os.path.join(self.model_path, "plots", "rate_vs_pt")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"rate_vs_pt_{nconst}const.png")
        plt.savefig(save_path)
        print(f"Rate vs pT plot saved to {save_path}")
        plt.show()
    """"

    
