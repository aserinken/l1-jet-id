import os
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

class FullJetDataV2:
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
    def __init__(self,
                 root,
                 nconst=32,
                 train=True,
                 datasets=None,
                 padding_value=0.0,
                 test_size=0.2,
                 random_state=42):

        self.root = Path(root)
        self.nconst = nconst
        self.train = train
        self.datasets = datasets if datasets else {}
        self.padding_value = padding_value
        self.test_size = test_size
        self.random_state = random_state

        # Jet-level and constituent-level features
        self.jet_features = [
            "jet_eta", "jet_phi", "jet_pt", "jet_pt_raw", "jet_mass", "jet_energy"
        ]
        self.constit_features = [
            "jet_pfcand_isfilled", "jet_pfcand_pt", "jet_pfcand_pt_rel", "jet_pfcand_pt_log",
            "jet_pfcand_eta", "jet_pfcand_phi", "jet_pfcand_mass", "jet_pfcand_energy",
            "jet_pfcand_puppiweight", "jet_pfcand_z0", "jet_pfcand_deta", "jet_pfcand_dphi",
            "jet_pfcand_isPhoton", "jet_pfcand_isElectron", "jet_pfcand_isElectronPlus",
            "jet_pfcand_isElectronMinus", "jet_pfcand_isMuon", "jet_pfcand_isMuonPlus",
            "jet_pfcand_isMuonMinus", "jet_pfcand_isNeutralHadron",
            "jet_pfcand_isChargedHadron", "jet_pfcand_isChargedHadronPlus",
            "jet_pfcand_isChargedHadronMinus"
        ]

        # Final data arrays
        self.x = None
        self.x_top = None
        self.y = None

        # Load and process data
        if self.train:
            self._load_all()
            self._combine()
            self._plot_feature_histograms()
        else:
            raise NotImplementedError("Test mode loading not implemented yet.")

    def _load_all(self):
        """Load all signal and background ROOT files into memory."""
        self.x_sig, self.x_top_sig, self.y_sig = self._load_class(self.datasets.get("sig", []), 1)
        self.x_bkg, self.x_top_bkg, self.y_bkg = self._load_class(self.datasets.get("bkg", []), 0)

    def _load_class(self, file_list, label):
        """Load ROOT files of a specific class (signal or background)."""
        x_all, x_top_all, y_all = [], [], []

        for path in file_list:
            print(f"[DEBUG] Opening file: {path}")
            with uproot.open(path)["outnano/Jets"] as tree:
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

        return x, x_top, y

    def _combine(self):
        """Combine signal and background data."""
        self.x = np.concatenate([self.x_bkg, self.x_sig], axis=0)
        self.x_top = np.concatenate([self.x_top_bkg, self.x_top_sig], axis=0)
        self.y = np.concatenate([self.y_bkg, self.y_sig], axis=0)

        print("[FullJetDataV2] Combined x shape:", self.x.shape)
        print("[FullJetDataV2] Combined x_top shape:", self.x_top.shape)
        print("[FullJetDataV2] Combined y shape:", self.y.shape)

    def show_details(self):
        print("[FullJetDataV2] x shape:", self.x.shape)
        print("[FullJetDataV2] x_top shape:", self.x_top.shape)
        print("[FullJetDataV2] y shape:", self.y.shape)
        print("[FullJetDataV2] Jet features:", self.jet_features)
        print("[FullJetDataV2] Constituent features:", self.constit_features)
        print("[FullJetDataV2] nconst:", self.nconst)
        print("[FullJetDataV2] padding value:", self.padding_value)

    def _plot_feature_histograms(self):
        """Plot and save histograms for top-level and first-constituent features."""
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
            axs_const[i].hist(self.x[:, 0, i], bins=50, alpha=0.7)
            axs_const[i].set_title(f"First constituent: {feature}")
        fig_const.tight_layout()
        const_path = self.root / "constituent_features.png"
        fig_const.savefig(const_path)
        print(f"[INFO] Saved constituent histograms to {const_path}")
        plt.close(fig_const)



def main():
    # Example usage
    root_dir = "/work/aserinke/l1-jet-id/test/SC8Data"
    datasets={
        "sig": ["/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/lroberts/jetTaggingInputs/lightHbb_M20to80_Pt50to200_1745321368/data/lightH.root"],
        "bkg": ["/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/lroberts/jetTaggingInputs/SingleNeutrino_PU200_1746011721/data/singneut.root"]
    }

    data = FullJetDataV2(
        root=root_dir,
        nconst=32,
        train=True,
        datasets=datasets,
        padding_value=0.0,
        test_size=0.2,
        random_state=42
    )

    data.show_details()
