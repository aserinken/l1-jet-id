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
    Loader for flat jet ROOT samples that contain BOTH signal- and background-jets
    differentiated by `jet_genmatch_Nprongs` (2 → signal, 0 → background).

    The label array `y` is shape (N, 1) with values 0 or 1.
    """

    # -------------------------------------------------------- INIT ----
    def __init__(
        self,
        root: str,
        nconst: int = 32,
        train: bool = True,
        datasets: dict | None = None,         
        padding_value: float = 0.0,
        test_size: float = 0.2,
        random_state: int = 42,
        kfolds: int = 0,
        nprong_branch: str = "jet_genmatch_Nprongs",
    ):
        super().__init__()
        self.root = Path(root)
        self.nconst = nconst
        self.train = train
        self.datasets = datasets or {}
        self.padding_value = padding_value
        self.test_size = test_size
        self.random_state = random_state
        self.kfolds = kfolds
        self.seed = random_state
        self.nprong_branch = nprong_branch

        self.output_dir = self.root / "SC8Data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # feature lists -------------------------------------------------
        self.jet_features = [
            "jet_pt_phys", "jet_eta_phys", "jet_phi_phys", "jet_mass"
        ]
        self.constit_features = [
            "jet_pfcand_pt", "jet_pfcand_pt_rel", "jet_pfcand_pt_log",
            "jet_pfcand_deta", "jet_pfcand_dphi", "jet_pfcand_mass",
            "jet_pfcand_isPhoton", "jet_pfcand_isElectronPlus",
            "jet_pfcand_isElectronMinus", "jet_pfcand_isMuonPlus",
            "jet_pfcand_isMuonMinus", "jet_pfcand_isNeutralHadron",
            "jet_pfcand_isChargedHadronMinus", "jet_pfcand_isChargedHadronPlus",
            "jet_pfcand_z0", "jet_pfcand_dxy", "jet_pfcand_isfilled",
            "jet_pfcand_puppiweight", "jet_pfcand_emid", "jet_pfcand_quality"
        ]

        # public containers --------------------------------------------
        self.x = self.x_top = self.y = None
        self.kfold_indices = None
        self.njets = self.nfeats = None

        # ------------------------------------------------ LOAD PIPELINE
        if self.train:
            self._load_all()          # fills self.x, self.x_top, self.y
            self._split_train_test()
            self._plot_feature_histograms()
            self.save_split_data()
            if self.kfolds > 0:
                self._split_kfold()
        else:
            self.load_split_data()

        self.njets = self.x.shape[0]
        self.nfeats = self.x.shape[-1]

    # ----------------------------------------------- ROOT → numpy -----
    def _load_all(self):
        """Read every ROOT file once and build (x, x_top, y) with scalar labels."""
        xs, x_tops, ys = [], [], []

        file_list = self.datasets.get("all", [])
        if not file_list:
            raise ValueError("datasets['all'] must list at least one ROOT file.")

        for path in file_list:
            print(f"[DEBUG] Opening {path}")
            with uproot.open(path)["outnano/Jets;1"] as tree:
                if tree.num_entries == 0:
                    print("[DEBUG]   0 entries → skipped")
                    continue

                branches = (
                    self.jet_features +
                    self.constit_features +
                    [self.nprong_branch]
                )
                arr = tree.arrays(branches, library="ak")

                # jet-level tensor --------------------------------------
                x_top = np.stack(
                    [ak.to_numpy(arr[f]) for f in self.jet_features], axis=-1
                )

                # constituent tensor ------------------------------------
                cons = ak.zip({f: arr[f] for f in self.constit_features})
                cons = ak.pad_none(cons, self.nconst, clip=True)
                cons = ak.fill_none(
                    cons, {k: self.padding_value for k in self.constit_features}
                )
                cons = ak.to_regular(cons)
                x_const = np.stack(
                    [ak.to_numpy(cons[f]) for f in cons.fields], axis=-1
                )

                # labels -------------------------------------------------
                npr = ak.to_numpy(arr[self.nprong_branch])
                mask_sig = npr == 2
                mask_bkg = npr == 0

                if mask_sig.any():
                    xs.append(x_const[mask_sig])
                    x_tops.append(x_top[mask_sig])
                    ys.append(np.ones(mask_sig.sum(), dtype=np.int8))

                if mask_bkg.any():
                    xs.append(x_const[mask_bkg])
                    x_tops.append(x_top[mask_bkg])
                    ys.append(np.zeros(mask_bkg.sum(), dtype=np.int8))

                print(f"[DEBUG]   kept {mask_sig.sum()+mask_bkg.sum()} jets")

        self.x      = np.concatenate(xs,      axis=0)
        self.x_top  = np.concatenate(x_tops,  axis=0)
        self.y      = np.concatenate(ys,      axis=0)[:, None]   # (N,1)

        sig = int(self.y.sum())
        bkg = self.y.shape[0] - sig
        print(f"[INFO] Loaded total jets: {self.y.shape[0]}  (sig={sig}, bkg={bkg})")

    # ------------------------------------------------ train/test split
    def _split_train_test(self):
        if self.test_size <= 0:
            return
        (self.x_train, self.x_test,
         self.x_top_train, self.x_top_test,
         self.y_train, self.y_test) = train_test_split(
            self.x, self.x_top, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y.ravel()          # ← scalar labels
        )
        if self.train:
            self.x, self.x_top, self.y = self.x_train, self.x_top_train, self.y_train
            tag = "Training"
        else:
            self.x, self.x_top, self.y = self.x_test, self.x_top_test, self.y_test
            tag = "Test"
        print(f"[INFO] {tag} shapes: x={self.x.shape}, y={self.y.shape}")

    # ------------------------------------------------ K-fold split
    def _split_kfold(self):
        print(f"[INFO] {self.kfolds}-fold CV split …")
        skf = StratifiedKFold(
            n_splits=self.kfolds, shuffle=True, random_state=self.seed
        )
        self.kfold_indices = list(skf.split(self.x, self.y.ravel()))

    # ------------------------------------------------ plots (optional)
    def _plot_feature_histograms(self):
        fig, axs = plt.subplots(len(self.jet_features), 1,
                                figsize=(6, 2 * len(self.jet_features)))
        for i, f in enumerate(self.jet_features):
            axs[i].hist(self.x_top[:, i], bins=50, alpha=0.7)
            axs[i].set_title(f)
        fig.tight_layout()
        fig.savefig(self.output_dir / "top_level_features.png")
        plt.close(fig)

        fig, axs = plt.subplots(len(self.constit_features), 1,
                                figsize=(6, 2 * len(self.constit_features)))
        for i, f in enumerate(self.constit_features):
            axs[i].hist(self.x[:, :, i].ravel(), bins=50, alpha=0.7, log=True)
            axs[i].set_title(f)
        fig.tight_layout()
        fig.savefig(self.output_dir / "constituent_features.png")
        plt.close(fig)

    # ------------------------------------------------ save / load
    def save_split_data(self):
        np.save(self.output_dir / f"proc_train_{self.const}const.npy",     self.x_train)
        np.save(self.output_dir / f"proc_top_train_{self.const}const.npy", self.x_top_train)
        np.save(self.output_dir / f"proc_top_labels_train_{self.const}const.npy", self.y_train)
        np.save(self.output_dir / f"proc_labels_train_{self.const}const.npy",     self.y_train)
        np.save(self.output_dir / f"proc_test_{self.const}const.npy",      self.x_test)
        np.save(self.output_dir / f"proc_top_test_{self.const}const.npy",  self.x_top_test)
        np.save(self.output_dir / f"proc_top_labels_test_{self.const}const.npy",  self.y_test)
        np.save(self.output_dir / f"proc_labels_test_{self.const}const.npy",      self.y_test)

    def load_split_data(self):
        subset = "train" if self.train else "test"
        self.x     = np.load(self.output_dir / f"proc_{subset}_{self.const}const.npy")
        self.x_top = np.load(self.output_dir / f"proc_top_{subset}_{self.const}const.npy")
        self.y     = np.load(self.output_dir / f"proc_labels_{subset}_{self.const}const.npy")
        print(f"[INFO] Loaded {subset} arrays: x={self.x.shape}, x_top={self.x_top.shape}, y={self.y.shape}")

    # ------------------------------------------------ quick info
    def show_details(self):
        print("x shape :", self.x.shape)
        print("y shape :", self.y.shape)
        print("njets   :", self.njets)
        print("nfeats  :", self.nfeats)
