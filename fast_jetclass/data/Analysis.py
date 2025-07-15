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
        self.dir_path = self.model_path / "data" / "test" / "SC8Data"
        
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
            model_dir = model_root / f"SC8_1node_{n}const_synth" / "kfolding1"
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
        signal_scores = y_pred_probs.flatten()  # shape = (numSamples,)

        # y_test_3d is one-hot (e.g. [[1,0],[0,1],...]),
        # convert it to integer labels [0 or 1]
        y_true = y_test_3d.flatten()

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
            y_true = y_test_3d.flatten()  # Assuming y_test_3d is already in binary format

            # Load the corresponding model
            all_models = self.get_paths_models()
            model = self.load_model_for_nconst(nconst, all_models)
            if model is None:
                print(f"Skipping nconst={nconst}: no model loaded.")
                continue

            # Generate predictions
            y_pred_probs = model.predict(x_test_3d)  
            signal_scores = y_pred_probs.flatten()  # shape = (numSamples,)

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
        y_true = data["3d_labels"].flatten()
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
        # Extract probability of the "signal" class (assumed at column index 1)
        signal_scores = y_pred_probs.flatten()
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
        y_true_base = (base_data["3d_labels"]).flatten()
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
            y_true = data["3d_labels"].flatten()
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
            signal_scores = y_pred_probs.flatten()
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

    def plot_output_dist_model(
            self,
            nconst: int = 8,
            pt_cut: float = 0,
            fixed_eta: float = 2.4,
            fixed_mass: float = 0):
        """
        Plot model-score distributions for signal and background after a
        kinematic pre-selection.

        Args
        ----
        nconst      : which model (constituent count) to load
        pt_cut      : leading-jet pT threshold [GeV]
        fixed_eta   : |η| window
        fixed_mass  : jet mass threshold [GeV]
        """

        import numpy as np, matplotlib.pyplot as plt, tensorflow as tf, os

        # ---------------- data & model ------------------------------------
        data = self.load_data(nconst)
        if any(data[k] is None for k in ("3d_data", "2d_data", "3d_labels")):
            print(f"Missing tensors for nconst={nconst}.");  return

        twoD   = data["2d_data"]
        x3d    = data["3d_data"]
        y_true = data["3d_labels"].flatten()     # 0 = bkg , 1 = sig

        # kinematic mask (same cuts you had)
        mask = (twoD[:, 0] >= pt_cut) \
            & (np.abs(twoD[:, 1]) < fixed_eta) \
            & (twoD[:, 3] >= fixed_mass)

        if mask.sum() == 0:
            print("No events survive the selection.");  return

        model = self.load_model_for_nconst(nconst)
        if model is None:
            print("Model not found.");  return

        scores = model.predict(x3d).flatten()  # shape = (numSamples,)
        scores_sel = scores[mask]
        y_sel      = y_true[mask]

        sig_scores = scores_sel[y_sel == 1]
        bkg_scores = scores_sel[y_sel == 0]

        # ---------------- plot --------------------------------------------
        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, 1, 50)

        plt.hist(bkg_scores, bins=bins, density=True,
                histtype="stepfilled", color="tab:orange", alpha=0.6,
                label=f"background  (n={len(bkg_scores)})")
        plt.hist(sig_scores, bins=bins, density=True,
                histtype="step", lw=2, color="tab:blue",
                label=f"signal      (n={len(sig_scores)})")

        plt.xlabel("Model Score")
        plt.ylabel("Normalized density")
        #Make y axis logarithmic
        plt.yscale("log")
        plt.title(f"Score distributions nconst={nconst},  "
                f"pT>{pt_cut} GeV, |η|<{fixed_eta}, mass>{fixed_mass} GeV")
        plt.legend()
        plt.grid(ls="--", lw=0.4)

        # ---------------- save & show -------------------------------------
        out_dir = os.path.join(self.model_path, "plots", "output_distribution")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir,
                f"score_dist_n{nconst}_pt{pt_cut}_m{fixed_mass}.png")
        plt.savefig(fname);  print("Saved →", fname)
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
        y_true_base = (base_data["3d_labels"]).flatten()
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
    
    

    def rate_vs_ptcuts(
        self,
        nconst=16,
        fixed_eta=2.4,
        fixed_mass=0,
        total_event_rate=30845,          
        pt_thresholds=None,
        decision_thresholds=None
        ):

        import numpy as np, matplotlib.pyplot as plt, tensorflow as tf, os

        if pt_thresholds is None:
            pt_thresholds = np.arange(0, 300, 1)
        if decision_thresholds is None:
            decision_thresholds = np.array([0.7, 0.8, 0.9, 0.95, 0.99])

        target_rates_khz = [58]        
        ref_ptcuts       = [50, 70, 80, 100, 110, 126]  

        # --------------------------------------------------------------------
        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"No model loaded for nconst={nconst}.")
            return

        # --------------------------------------------------------------------
        data = self.load_data(nconst)
        if data["3d_labels"] is None or data["2d_data"] is None:
            print("Missing data – aborting.")
            return

        twoD = data["2d_data"]                          
        n_events = len(twoD)
        y_true = data["3d_labels"].flatten()
        bkg_mask_all = (y_true == 0)    
        sig_mask_all = (y_true == 1)                
        n_bkg_events   = bkg_mask_all.sum() 
        n_sig_events   = sig_mask_all.sum()  

        # compute scores
        scores = model.predict(data["3d_data"]).flatten()

        # -------------------------------------------------------------------- #
        # baseline curve (rate vs pT)                                           #
        # -------------------------------------------------------------------- #
        baseline_rate = []
        baseline_eff  = []
        for thr in pt_thresholds:
            kin_mask = (twoD[:, 0] >= thr) & (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass)
            evt_pass = kin_mask & bkg_mask_all 
            sig_pass = kin_mask & sig_mask_all         
            rate     = evt_pass.sum() / n_bkg_events * total_event_rate
            eff      = sig_pass.sum() / n_sig_events
            baseline_rate.append(rate)
            baseline_eff.append(eff)

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.plot(pt_thresholds, baseline_rate, "--o", color="tab:blue", label="Baseline Rate")
        ax2 = ax1.twinx()
        ax2.plot(pt_thresholds, baseline_eff, "-o", color="tab:blue", label="Baseline Efficiency")

        # -------------------------------------------------------------------- #
        # tagger curves                                                         #
        # -------------------------------------------------------------------- #
        colours = ["tab:green", "tab:orange", "tab:olive", "tab:brown", "tab:purple"]
        tagger_curves = {}
        for dt, col in zip(decision_thresholds, colours):
            nn_pass = scores > dt                         
            tagger_rate = []
            tagger_eff  = []
            for thr in pt_thresholds:
                kin_mask = (twoD[:, 0] >= thr) & (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= fixed_mass)
                evt_pass = kin_mask & nn_pass & bkg_mask_all
                sig_pass = kin_mask & nn_pass & sig_mask_all              
                rate     = evt_pass.sum() / n_bkg_events * total_event_rate
                eff      = sig_pass.sum() / n_sig_events
                tagger_rate.append(rate)
                tagger_eff.append(eff)

            ax1.plot(pt_thresholds, tagger_rate, "--", color=col, label=f"Tagger Rate (dt>{dt})")
            ax2.plot(pt_thresholds, tagger_eff, "-", color=col, label=f"Tagger Efficiency (dt>{dt})")

            tagger_curves[dt] = (tagger_rate, tagger_eff)

        # -------------------------------------------------------------------- #
        # dressing                                                             #
        # -------------------------------------------------------------------- #
        ax1.set_xlabel("Top-level $p_{T}$ cut [GeV]")
        ax1.set_ylabel("Trigger Rate [kHz]")
        ax1.set_xlim(80, 150)
        ax2.set_xlim(80, 150)
        ax1.set_ylim(0, 280)
        ax2.set_ylim(0.45,0.9)
        ax2.set_ylabel("Signal Efficiency")
        plt.title("Rate vs $p_{T}$ and Signal Efficiency with" f" eta < {fixed_eta} and mass > {fixed_mass}")
        plt.grid(True, ls="--", lw=0.5)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # save
        out_dir = os.path.join(self.model_path, "plots", "rate_vs_ptcuts")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"rate_vs_ptcuts_n{nconst}_mass{fixed_mass}.png")
        plt.savefig(fname);  print("Saved →", fname)
        plt.show()


        # ------------------------------------------------------------------ #
        # summary tables                                                     #
        # ------------------------------------------------------------------ #
        from numpy import array, interp

        def interp_at(x, y, xstar):
            """Linear interpolation with graceful NaN outside range."""
            x, y = array(x), array(y)
            if (xstar < x.min()) or (xstar > x.max()):
                return np.nan
            # np.interp expects ascending x; your rate arrays are descending
            idx = np.argsort(x)
            return interp(xstar, x[idx], y[idx])

        print("\n===  Efficiency at target trigger-rate budgets (interpolated) ===")
        for R in target_rates_khz:
            eff_base  = interp_at(baseline_rate, baseline_eff,  R)
            pt_base   = interp_at(baseline_rate, pt_thresholds, R)
            if np.isnan(eff_base):
                print(f"\nRate {R} kHz is outside curve range — skipped.");  continue
            print(f"\nTarget rate  {R:.1f} kHz  (baseline pT ≈ {pt_base:.1f} GeV)")
            print(f"  baseline    :  eff = {eff_base:.3f}")
            for dt in decision_thresholds:
                rate_vec, eff_vec = tagger_curves[dt]
                eff_t   = interp_at(rate_vec, eff_vec, R)
                pt_t    = interp_at(rate_vec, pt_thresholds, R)
                if np.isnan(eff_t):
                    print(f"  score>{dt:<5}:  — out of range —")
                else:
                    print(f"  score>{dt:<5}:  eff = {eff_t:.3f}  "
                        f"(pT ≈ {pt_t:.1f} GeV)")

        print("\n===  Rate reduction & efficiency at fixed pT cuts (nearest) ===")
        def nearest_idx(arr, val):                 
            return int(np.argmin(np.abs(array(arr) - val)))

        for pt_ref in ref_ptcuts:
            i = nearest_idx(pt_thresholds, pt_ref)
            print(f"\npT ≈ {pt_thresholds[i]:.1f} GeV")
            print(f"  baseline    :  rate = {baseline_rate[i]:.1f} kHz,  "
                f"eff = {baseline_eff[i]:.3f}")
            for dt in decision_thresholds:
                rate_vec, eff_vec = tagger_curves[dt]
                r   = rate_vec[i];   e = eff_vec[i]
                print(f"  score>{dt:<5}:  rate = {r:6.1f} kHz  "
                    f"({r/baseline_rate[i]:.2f}×),  "
                    f"eff = {e:.3f}  ({e/baseline_eff[i]:.2f}×)")
                
        # ------------------------------------------------------------------ #
        #  one scatter plot per budget:  pT cut  vs  efficiency              #
        # ------------------------------------------------------------------ #
        from matplotlib.ticker import MaxNLocator

        pad_frac = 0.05           # 5 % padding around min/max

        for R in target_rates_khz:
            # baseline
            pt_b  = interp_at(baseline_rate, pt_thresholds, R)
            eff_b = interp_at(baseline_rate, baseline_eff,   R)
            if np.isnan(pt_b):
                print(f"Rate {R} kHz outside baseline curve – skipped plot.");  continue

            figR, axR = plt.subplots(figsize=(8, 6))

            # keep lists of the points we actually plot --------------
            pts_x, pts_y = [pt_b], [eff_b]

            axR.scatter(pt_b, eff_b, marker="*", s=120,
                        color="tab:blue", label="baseline")

            # tagger circles
            for dt, col in zip(decision_thresholds, colours):
                rate_vec, eff_vec = tagger_curves[dt]
                pt_t  = interp_at(rate_vec, pt_thresholds, R)
                eff_t = interp_at(rate_vec, eff_vec,        R)
                if np.isnan(pt_t):          # tagger never reaches R kHz
                    continue
                pts_x.append(pt_t);  pts_y.append(eff_t)

                axR.scatter(pt_t, eff_t, marker="o", s=70,
                            color=col, label=f"score>{dt}")

            # ---------- automatic limits ---------------------------
            x_lo, x_hi = min(pts_x), max(pts_x)
            y_lo, y_hi = min(pts_y), max(pts_y)

            # symmetric 5 % padding
            axR.set_xlim(x_lo - pad_frac*(x_hi-x_lo),  x_hi + pad_frac*(x_hi-x_lo))
            axR.set_ylim(max(0, y_lo - pad_frac*(y_hi-y_lo)),  1.01*y_hi)
            # -------------------------------------------------------

            axR.set_xlabel(r"$p_T$ cut  [GeV]")
            axR.set_ylabel("Signal efficiency")
            axR.set_title(f"Trigger Rate ≈ {R} kHz  "
                        f"($η<{fixed_eta}$, mass > {fixed_mass} GeV)")
            axR.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
            axR.grid(True, ls="--", lw=0.4)
            axR.legend(fontsize=8, loc="lower left")

            # save
            fnameR = os.path.join(out_dir,
                    f"pt_vs_eff_at_{int(R)}kHz_n{nconst}_m{fixed_mass}.png")
            plt.savefig(fnameR);  print("Saved →", fnameR)
            plt.show()
        
    def rate_vs_masscuts(
        self,
        nconst=16,
        fixed_eta=2.4,
        total_event_rate=30845,          
        mass_thresholds=None,
        decision_thresholds=None
        ):

        import numpy as np, matplotlib.pyplot as plt, tensorflow as tf, os

        if mass_thresholds is None:
            mass_thresholds = np.arange(0, 100, 1)
        if decision_thresholds is None:
            decision_thresholds = np.array([0.5, 0.8, 0.9, 0.95, 0.99])

        target_rates_khz = [58]        
        ref_masscuts = [5, 10, 15, 20, 25, ]  

        # --------------------------------------------------------------------
        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"No model loaded for nconst={nconst}.")
            return

        # one pass over the data ------------------------------------------------
        data = self.load_data(nconst)
        if data["3d_labels"] is None or data["2d_data"] is None:
            print("Missing data – aborting.")
            return

        twoD = data["2d_data"]                          
        n_events = len(twoD)
        y_true = data["3d_labels"].flatten()
        bkg_mask_all = (y_true == 0)    
        sig_mask_all = (y_true == 1)               
        n_bkg_events   = bkg_mask_all.sum() 
        n_sig_events   = sig_mask_all.sum()  

        # tagger scores
        scores = model.predict(data["3d_data"]).flatten()

        # -------------------------------------------------------------------- #
        # baseline curve (rate vs pT)                                           #
        # -------------------------------------------------------------------- #
        baseline_rate = []
        baseline_eff  = []
        for mass in mass_thresholds:
            kin_mask = (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= mass)
            evt_pass = kin_mask & bkg_mask_all 
            sig_pass = kin_mask & sig_mask_all         
            rate     = evt_pass.sum() / n_bkg_events * total_event_rate
            eff      = sig_pass.sum() / n_sig_events
            baseline_rate.append(rate)
            baseline_eff.append(eff)

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.semilogy(mass_thresholds, baseline_rate, "--o", color="tab:blue", label="Baseline Rate")
        ax2 = ax1.twinx()
        ax2.plot(mass_thresholds, baseline_eff, "-o", color="tab:blue", label="Baseline Efficiency")

        # -------------------------------------------------------------------- #
        # tagger curves                                                         #
        # -------------------------------------------------------------------- #
        colours = ["tab:green", "tab:orange", "tab:olive", "tab:brown", "tab:purple"]
        tagger_curves = {}
        for dt, col in zip(decision_thresholds, colours):
            nn_pass = scores > dt                        
            tagger_rate = []
            tagger_eff  = []
            for mass in mass_thresholds:
                kin_mask =  (np.abs(twoD[:, 1]) < fixed_eta) & (twoD[:, 3] >= mass)
                evt_pass = kin_mask & nn_pass & bkg_mask_all
                sig_pass = kin_mask & nn_pass & sig_mask_all              
                rate     = evt_pass.sum() / n_bkg_events * total_event_rate
                eff      = sig_pass.sum() / n_sig_events
                tagger_rate.append(rate)
                tagger_eff.append(eff)

            ax1.semilogy(mass_thresholds, tagger_rate, "--", color=col, label=f"Tagger Rate (dt>{dt})")
            ax2.plot(mass_thresholds, tagger_eff, "-", color=col, label=f"Tagger Efficiency (dt>{dt})")

            tagger_curves[dt] = (tagger_rate, tagger_eff)

        # -------------------------------------------------------------------- #
        # dressing                                                             #
        # -------------------------------------------------------------------- #
        ax1.set_xlabel("Top-level mass cut [GeV]")
        ax1.set_ylabel("Trigger Rate [kHz]")
        ax2.set_ylabel("Signal Efficiency")
        ax1.set_xlim(0, 40)
        ax2.set_ylim(0.4, 1)
        plt.title("Rate vs mass and Signal Efficiency with" f" eta < {fixed_eta}")
        plt.grid(True, ls="--", lw=0.5)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # save
        out_dir = os.path.join(self.model_path, "plots", "rate_vs_masscuts")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"rate_vs_masscuts_n{nconst}.png")
        plt.savefig(fname);  print("Saved →", fname)
        plt.show()

        from numpy import array, interp

        def interp_at(x, y, xstar):
            """Linear interpolation with graceful NaN outside range."""
            x, y = array(x), array(y)
            if (xstar < x.min()) or (xstar > x.max()):
                return np.nan
            # np.interp expects ascending x; your rate arrays are descending
            idx = np.argsort(x)
            return interp(xstar, x[idx], y[idx])

        print("\n===  Efficiency at target trigger-rate budgets (interpolated) ===")
        for R in target_rates_khz:
            eff_base  = interp_at(baseline_rate, baseline_eff,  R)
            mass_base   = interp_at(baseline_rate, mass_thresholds, R)
            if np.isnan(eff_base):
                print(f"\nRate {R} kHz is outside curve range — skipped.");  continue
            print(f"\nTarget rate  {R:.1f} kHz  (baseline mass ≈ {mass_base:.1f} GeV)")
            print(f"  baseline    :  eff = {eff_base:.3f}")
            for dt in decision_thresholds:
                rate_vec, eff_vec = tagger_curves[dt]
                eff_t   = interp_at(rate_vec, eff_vec, R)
                mass    = interp_at(rate_vec, mass_thresholds, R)
                if np.isnan(eff_t):
                    print(f"  score>{dt:<5}:  — out of range —")
                else:
                    print(f"  score>{dt:<5}:  eff = {eff_t:.3f}  "
                        f"(mass ≈ {mass:.1f} GeV)")

        print("\n===  Rate reduction & efficiency at fixed mass cuts (nearest) ===")
        def nearest_idx(arr, val):
            return int(np.argmin(np.abs(array(arr) - val)))

        for mass_ref in ref_masscuts:
            i = nearest_idx(mass_thresholds, mass_ref)
            print(f"\nmass ≈ {mass_thresholds[i]:.1f} GeV")
            print(f"  baseline    :  rate = {baseline_rate[i]:.1f} kHz,  "
                f"eff = {baseline_eff[i]:.3f}")
            for dt in decision_thresholds:
                rate_vec, eff_vec = tagger_curves[dt]
                r   = rate_vec[i];   e = eff_vec[i]
                print(f"  score>{dt:<5}:  rate = {r:6.1f} kHz  "
                    f"({r/baseline_rate[i]:.2f}×),  "
                    f"eff = {e:.3f}  ({e/baseline_eff[i]:.2f}×)")



    def compare_henry_with_mine(
        self,
        nconst=8,
    ):
        """
        Load the inputs from henry and run inference on them and see the output
        """
        herny_path = "/work/aserinke/SC8_Data/Henry_Data/inputs_physical.npy"
        output_path = "/work/aserinke/SC8_Data/Henry_Data/outputs_physical.npy"
        import numpy as np, tensorflow as tf, os
        from tensorflow.keras.models import load_model
        #Load
        henry_data = np.load(herny_path)
        henry_data = henry_data.astype(np.float32)
        print("Loaded Henry data keys:", henry_data.shape)

        path = "/work/aserinke/l1-jet-id/scripts/trained_deepsets/SC8SCAN_deepsets_8bit_8const_synth/kfolding1"
        model_quant = load_model(path, compile=False)

        #Load model
        model = self.load_model_for_nconst(nconst)
        if model is None:
            print(f"No model loaded for nconst={nconst}.")
            return
        #Predict
        henry_scores = model.predict(henry_data)
        henry_scores = henry_scores.astype(np.float32)
        #save the scores
        henry_scores_path = os.path.join(self.model_path, f"scores_n{nconst}.npy")
        np.save(henry_scores_path, henry_scores)
        print("Saved Henry scores to:", henry_scores_path)
        print("Standalone scores shape:", henry_scores.shape)

        scores_quant = model_quant.predict(henry_data)
        scores_quant = scores_quant.astype(np.float32)

        #Load Henry outputs
        henry_outputs = np.load(output_path)
        henry_outputs = henry_outputs.astype(np.float32)
        print("Emulator outputs shape:", henry_outputs.shape)
        if henry_scores.shape != henry_outputs.shape:
            print("Shapes do not match! Standalone scores shape:", henry_scores.shape, 
                  "Emulator outputs shape:", henry_outputs.shape)
            return

        #Compare
        for col, label in zip([0, 1], ["Background", "Signal"]):
            print(f"\nComparing {label} probabilities:")
            comparisons = np.isclose(henry_scores[:, col], henry_outputs[:, col], atol=0.05)
            num_matches = np.sum(comparisons)
            num_mismatches = np.sum(~comparisons)
            mean_diff = np.mean(np.abs(henry_scores[:, col] - henry_outputs[:, col]))
            print(f"  Number of matches: {num_matches}, mismatches: {num_mismatches}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")

        overall_diff = np.mean(np.abs(henry_scores - henry_outputs))
        print(f"\nOverall mean absolute difference: {overall_diff:.6f}")

        for col, label in zip([0, 1], ["Background", "Signal"]):
            print(f"\nComparing {label} probabilities:")
            comparisons = np.isclose(henry_scores[:, col], scores_quant[:, col], atol=0.01)
            num_matches = np.sum(comparisons)
            num_mismatches = np.sum(~comparisons)
            mean_diff = np.mean(np.abs(henry_scores[:, col] - scores_quant[:, col]))
            print(f"  Number of matches: {num_matches}, mismatches: {num_mismatches}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")

        overall_diff = np.mean(np.abs(henry_scores - scores_quant))
        print(f"\nOverall mean absolute difference: {overall_diff:.6f}")


        #first 5 events for both scores and outputs.
        print("\nFirst 10 Standalone scores events:")
        print(henry_scores[:10])
        print("\nFirst 10 Emulator events:")
        print(henry_outputs[:10])

        # Compute absolute differences for each class
        diff_background = np.abs(henry_scores[:, 0] - henry_outputs[:, 0])
        diff_signal = np.abs(henry_scores[:, 1] - henry_outputs[:, 1])

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # Histogram for Background differences
        plt.subplot(1, 2, 1)
        plt.hist(diff_background, bins=50, color="skyblue", edgecolor="k", alpha=0.7)
        plt.xlabel("|Standalone - Emulator|")
        plt.ylabel("Frequency")
        plt.title("0th Node Difference Histogram")

        # Histogram for Signal differences
        plt.subplot(1, 2, 2)
        plt.hist(diff_signal, bins=50, color="salmon", edgecolor="k", alpha=0.7)
        plt.xlabel("|Standalone - Emulator|")
        plt.title("1st Node Difference Histogram")


        plt.tight_layout()
        plt.show()

        # After computing and plotting your histograms (e.g. for differences)
        import os
        out_dir = os.path.join(self.model_path, "plots", "compare_henry_diffs")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"diff_histograms_n{nconst}.png")
        plt.savefig(fname)
        print("Saved →", fname)
        plt.show()



    
