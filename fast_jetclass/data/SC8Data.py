import os
import numpy as np
import awkward as ak
import sklearn.model_selection
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import uproot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

class SC8Data(object):
    """
    A simplified data class for loading and processing flat jet ROOT data.

    This class supports padded constituent arrays per jet and extracts both jet-level and constituent-level features
    from a flat ROOT tree, eliminating the need for SC8-Puppi matching and custom PDG one-hot encoding.

    Args:
        root: Directory where raw ROOT files are located.
        nconst: Number of constituents per jet (padded or truncated).
        train: Whether to load training (True) or test (False) data.
        datasets: Dict with keys "bkg" and "sig" pointing to lists of ROOT file paths.
        padding_value: Value used to pad missing constituents.
        test_size: Fraction of data used for testing if splitting.
        random_state: Seed for reproducibility.
    """
    def __init__(
            self,
            root: str,
            nconst: int=32,
            train: bool=True,
            datasets: dict=None,
            padding_value: float=0.0,
            test_size: float=0.2,
            random_state: int=42,
            kfolds: int=0,
    ):
        super().__init__()
        self.root = Path(root)
        self.nconst = nconst
        self.train = train
        self.datasets = datasets if datasets else {}
        self.padding_value = padding_value
        self.test_size = test_size
        self.random_state = random_state
        self.kfolds = kfolds
        self.seed = random_state

        self.output_dir = self.root / "SC8Data"

        # Jet-level and constituent-level features
        self.jet_features = [
            "jet_eta", "jet_phi", "jet_pt", "jet_mass", "jet_energy"
        ]
        self.constit_features = [
            "jet_pfcand_pt", "jet_pfcand_pt_rel","jet_pfcand_pt_log",
            "jet_pfcand_deta", "jet_pfcand_dphi", "jet_pfcand_mass",
            "jet_pfcand_isPhoton", "jet_pfcand_isElectronPlus", "jet_pfcand_isElectronMinus", 
            "jet_pfcand_isMuonPlus", "jet_pfcand_isMuonMinus", "jet_pfcand_isNeutralHadron",
            "jet_pfcand_isChargedHadronMinus", "jet_pfcand_isChargedHadronPlus", "jet_pfcand_z0",
            "jet_pfcand_dxy", "jet_pfcand_isfilled", "jet_pfcand_puppiweight",
            "jet_pfcand_emid", "jet_pfcand_quality"
        ]

        # Final data arrays
        self.x = None
        self.x_top = None
        self.y = None
        self.kfold_indices = None

        # Set after loading
        self.njets = None
        self.nfeats = None

        # Load and process data
        # In __init__:
        if self.train:
            self._load_all()
            self._combine()
            self._split_train_test()
            self._plot_feature_histograms()
            self.save_split_data()
            #lets delete the original data to save memory
            del self.x_sig, self.x_top_sig, self.y_sig
            del self.x_bkg, self.x_top_bkg, self.y_bkg
            del self.x, self.x_top, self.y
            self.load_split_data()
            if self.kfolds > 0 and self.train:
                self._split_kfold()
            # Set counts
            self.njets = self.x.shape[0]
            self.nfeats = self.x.shape[-1]
        else:
            self.load_split_data()
            self.njets = self.x.shape[0]
            self.nfeats = self.x.shape[-1]

        
        self.x_top = None

    def _load_all(self):
        """Load all signal and background ROOT files into memory."""
        self.x_sig, self.x_top_sig, self.y_sig = self._load_class(self.datasets.get("sig", []), 1)
        self.x_bkg, self.x_top_bkg, self.y_bkg = self._load_class(self.datasets.get("bkg", []), 0)
        print("[FullJetDataV2] Loaded signal and background data:")

    def _load_class(self, file_list, label):
        """Load ROOT files of a specific class (signal or background)."""
        x_all, x_top_all, y_all = [], [], []

        for path in file_list:
            print(f"[DEBUG] Opening file: {path}")
            with uproot.open(path)["outnano/Jets;1"] as tree:
                if tree.num_entries == 0:
                    print(f"[DEBUG] Skipping file with 0 entries: {path}")
                    continue
                all_fields = self.jet_features + self.constit_features
                all_data = tree.arrays(all_fields, library="ak")
                print("[DEBUG] Loaded all fields:", all_data.fields)

                x_top = np.stack([ak.to_numpy(all_data[f]) for f in self.jet_features], axis=-1)
                x_top_all.append(x_top)

                pfcand_dict = {f: all_data[f] for f in self.constit_features}
                const_arr = ak.zip(pfcand_dict)
                const_arr = ak.pad_none(const_arr, self.nconst, clip=True)
                const_arr = ak.fill_none(const_arr, {key: self.padding_value for key in self.constit_features})
                const_arr = ak.to_regular(const_arr)

                stacked = np.stack([ak.to_numpy(const_arr[f]) for f in const_arr.fields], axis=-1)
                x_all.append(stacked)

                n_jets = len(all_data[self.jet_features[0]])
                y_all.append(np.full(n_jets, label))
                print(f"[DEBUG] Selected {n_jets} jets from file: {path}")

        if not x_all:
            raise ValueError(f"No usable data found for label {label}. All files may be empty or invalid.")

        x = np.concatenate(x_all, axis=0)
        x_top = np.concatenate(x_top_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        y = np.eye(2)[y]  # One-hot encoding

        max_events = 100000
        if x.shape[0] > max_events:
            print(f"[DEBUG] Limiting to first {max_events} events for label {label}.")
            idx = np.random.default_rng(self.random_state).choice(x.shape[0], max_events, replace=False)
            x = x[idx]
            x_top = x_top[idx]
            y = y[idx]
        
        return x, x_top, y

    def _combine(self):
        """Combine signal and background data."""
        self.x = np.concatenate([self.x_bkg, self.x_sig], axis=0)
        self.x_top = np.concatenate([self.x_top_bkg, self.x_top_sig], axis=0)
        self.y = np.concatenate([self.y_bkg, self.y_sig], axis=0)

        print("[FullJetDataV2] Combined x shape:", self.x.shape)
        print("[FullJetDataV2] Combined x_top shape:", self.x_top.shape)
        print("[FullJetDataV2] Combined y shape:", self.y.shape)

    def _split_train_test(self):
        """Split the combined data into training and test sets and store both splits."""
        if self.test_size > 0:
            from sklearn.model_selection import train_test_split
            (self.x_train, self.x_test, 
            self.x_top_train, self.x_top_test, 
            self.y_train, self.y_test) = train_test_split(
                self.x, self.x_top, self.y,
                test_size=self.test_size, random_state=self.random_state,
                stratify=np.argmax(self.y, axis=-1)
            )
            if self.train:
                self.x, self.x_top, self.y = self.x_train, self.x_top_train, self.y_train
                print("[FullJetDataV2] Training data shapes:", self.x.shape, self.x_top.shape, self.y.shape)
            else:
                self.x, self.x_top, self.y = self.x_test, self.x_top_test, self.y_test
                print("[FullJetDataV2] Test data shapes:", self.x.shape, self.x_top.shape, self.y.shape)

    def _split_kfold(self) -> None:
        """Split into k folds if requested."""
        if self.kfolds > 0 and self.train:
            print(f"[_split_kfold] Using {self.kfolds}-fold cross-validation...")
            skf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)
            self.kfold_indices = list(skf.split(self.x, np.argmax(self.y, axis=-1)))


    def show_details(self):
        print("[FullJetDataV2] x shape:", self.x.shape)
        print("[FullJetDataV2] y shape:", self.y.shape)
        print("[FullJetDataV2] Jet features:", self.jet_features)
        print("[FullJetDataV2] Constituent features:", self.constit_features)
        print("[FullJetDataV2] nconst:", self.nconst)
        print("[FullJetDataV2] padding value:", self.padding_value)
        print("[FullJetDataV2] njets:", self.njets)
        print("[FullJetDataV2] nfeats:", self.nfeats)

    def _plot_feature_histograms(self):
        """Plot and save histograms for top-level and all-constituent features."""

        fig_top, axs_top = plt.subplots(len(self.jet_features), 1, figsize=(6, len(self.jet_features)*2))
        for i, feature in enumerate(self.jet_features):
            axs_top[i].hist(self.x_top[:, i], bins=50, alpha=0.7)
            axs_top[i].set_title(f"Top-level feature: {feature}")
        fig_top.tight_layout()
        top_path = self.root / "top_level_features.png"
        fig_top.savefig(top_path)
        print(f"[INFO] Saved top-level histograms to {top_path}")
        plt.close(fig_top)

        fig_const, axs_const = plt.subplots(len(self.constit_features), 1, figsize=(6, len(self.constit_features)*2))
        for i, feature in enumerate(self.constit_features):
            axs_const[i].hist(self.x[:, :, i].flatten(), bins=50, alpha=0.7)
            axs_const[i].set_title(f"All constituent: {feature}")
            axs_const[i].set_yscale('log')
        fig_const.tight_layout()
        const_path = self.root / "constituent_features.png"
        fig_const.savefig(const_path)
        print(f"[INFO] Saved constituent histograms to {const_path}")
        plt.close(fig_const)

    def save_split_data(self, out_dir: str = None) -> None:
        """
        Save both the training and test data arrays to .npy files in a designated directory.
        If out_dir is not provided, a default folder 'saved_data' under self.root is used.
        """
        import os
        import numpy as np
        out_dir = self.output_dir
        if out_dir is None:
            out_dir = os.path.join(self.root, "saved_data")
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "train_x.npy"), self.x_train)
        np.save(os.path.join(out_dir, "train_x_top.npy"), self.x_top_train)
        np.save(os.path.join(out_dir, "train_y.npy"), self.y_train)
        np.save(os.path.join(out_dir, "test_x.npy"), self.x_test)
        np.save(os.path.join(out_dir, "test_x_top.npy"), self.x_top_test)
        np.save(os.path.join(out_dir, "test_y.npy"), self.y_test)

        print(f"[INFO] Saved training data to {os.path.join(out_dir, 'train_*.npy')}")
        print(f"[INFO] Saved test data to {os.path.join(out_dir, 'test_*.npy')}")

    def load_split_data(self, out_dir: str = None) -> None:
        """
        Load the test data arrays from .npy files in the designated directory.
        If out_dir is not provided, a default folder 'saved_data' under self.root is used.
        """
        import os
        import numpy as np

        out_dir = self.output_dir if self.output_dir is not None else os.path.join(self.root, "saved_data")
        if self.train:
            self.x = np.load(os.path.join(out_dir, "train_x.npy"))
            self.y = np.load(os.path.join(out_dir, "train_y.npy"))
        else:
            self.x = np.load(os.path.join(out_dir, "test_x.npy"))
            self.y = np.load(os.path.join(out_dir, "test_y.npy"))
        print(f"[INFO] Loaded test data shapes: x: {self.x.shape}, y: {self.y.shape}")


