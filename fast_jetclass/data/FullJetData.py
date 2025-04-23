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

class FullJetData(object):
    """
    Data Class for importing and processing full jet data.

    Raw data is available in /eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/lroberts/

    Args:
        root: The root directory of the data. It should contain a 'raw' and 'processed'
            folder with raw and processed data. Otherwise, these will be generated.
        nconst: Number of constituents to pad/truncate to.
        pdg_order: PDG codes to one-hot encode.
        norm: Chosen normalization.
        train: Whether this is training data or validation/test.
        kfolds: Number of splits for k-fold cross-validation.
        seed: Random seed for reproducibility.
        datasets: Dictionary with keys (e.g. "bkg", "sig") and values = list of ROOT paths.
        test_size: Fraction for train/test split.
    """

    def __init__(
        self,
        root: str,
        nconst: int = 32,
        pdg_order=None,
        norm: str = "standard",
        train: bool = True,
        kfolds: int = 0,
        seed: int = 42,
        datasets: dict = None,  # If needed
        test_size: float = 0.2,  # If needed
        random_state: int = 42,
        padding_value: float = 0,
    ):
        """
        Args:
            root: Directory to store/load the data.
            nconst: Number of constituents to pad/truncate to.
            pdg_order: PDG codes to one-hot encode.
            norm: Chosen normalization.
            train: Whether this is training data or validation/test.
            kfolds: Number of splits for k-fold cross-validation.
            seed: Random seed for reproducibility.
            datasets: Dictionary with keys (e.g. "bkg", "sig") and values = list of ROOT paths.
            test_size: Fraction for train/test split.
        """
        super().__init__()
        self.root = Path(root)
        self.data_dir = self.root
        self.preprocessed_path = self.root / "preprocessed"
        self.raw_path = self.root / "raw"
        self.processed = self.root / "processed"
        self.nconst = nconst
        self.train = train
        self.type = "train" if train else "val"
        self.norm = norm
        self.seed = seed
        self.kfolds = kfolds
        self.pdg_order = pdg_order if pdg_order else [22, -11, 11, -13, 13, 130, -211, 211]
        self.random_state = random_state
        self.padding_value = padding_value

        self.datasets = datasets if datasets else {}
        self.test_size = test_size

        self.proc_output_name = (f"{self.type}_{self.norm}_{self.nconst}const")
        self.preproc_output_name = (f"{self.nconst}const")
        # Final data
        self.x = None
        self.y = None

         # Internal placeholders
        self.sc8_bkg_ak = None
        self.sc8_sig_ak = None
        self.puppi_bkg_ak = None
        self.puppi_sig_ak = None

        #If we run the code for the val then we can just load the data from the preprocessed folder
        if self.train == False:
            print("Loading data for validation/test from preprocessed folder.")
            self.load_proc_test_data()
            return

        if self._check_preprocessed_data_exists()==True:
            print("Preprocessed data found. Loading from 'preprocessed' folder.")
            if self._check_processed_data_exists()== True:
                print("Processed data found. Loading from 'processed' folder.")
                self.load_normalized_data()
                self._split_kfold()
            else:
                print("Processed data not found. Loading from 'preprocessed' folder.")
                self.load_3d_data()
                #6) Combine bkg/sig => X, y
                self._combine_sig_bkg()
                #7) Normalize or split
                self._normalize_data()
                #8) K-fold splitting
                self._split_kfold()

        else:
            print("No preprocssed data found. Running full pipeline.")
            #1) Convert ROOT -> Parquet
            self._process_root_files()
            #2) Load sc8/puppi from Parquet
            self._load_parquet()
            #3) Match sc8 to Puppi
            self._match_puppi()
            #4) One-hot PDG
            self._encode_pdg()
            #5) Pad => 3D array
            self._pad_to_3d()
            self.load_3d_data()
            #6) Combine bkg/sig => X, y
            self._combine_sig_bkg()
            #7) Normalize or split
            self._normalize_data()
            #8) K-fold splitting
            self._split_kfold()

        self.njets = self.x.shape[0]
        self.nfeats = self.x.shape[-1]

    def _check_preprocessed_data_exists(self) -> bool:
        """
        Check if the preprocessed data directory exists.
        """
        proc_dir = self.preprocessed_path
        needed_files = [
            f"bkg_3d_{self.preproc_output_name}.npy",
            f"sig_3d_{self.preproc_output_name}.npy"]
        if not proc_dir.is_dir():
            return False
        return all((proc_dir / f).is_file() for f in needed_files)
    
    def _check_processed_data_exists(self) -> bool:
        """
        Check if the processed data directory exists.
        """
        proc_dir = self.processed
        needed_files = [
            f"proc_{self.proc_output_name}.npy",
            f"proc_labels_{self.proc_output_name}.npy"
        ]
        if not proc_dir.is_dir():
            return False
        return all((proc_dir / f).is_file() for f in needed_files)


    # ---------------------------------------------------------------------
    # A) Process all ROOT files -> Parquet
    # ---------------------------------------------------------------------
    def _process_root_files(self) -> None:
        """
        For each dataset in self.datasets, process its associated ROOT files.
        """
        for key, file_list in self.datasets.items():
            print(f"\n[ROOT->Parquet] Processing dataset: {key}")
            self._process_dataset(key, file_list)

    def _process_dataset(self, key, file_list) -> None:
        """
        Loop over a single dataset’s file list and convert each ROOT file to Parquet.
        """
        
        for i, root_path in enumerate(file_list):
            if not os.path.exists(root_path):
                print(f"  file not found: {root_path}, skipping.")
                continue
            self._process_single_file(key, i, root_path)

    def _process_single_file(self, key, file_index, root_path) -> None:
        """
        Convert one ROOT file (both sc8 and puppi) into Parquet.
        """
        print(f"  Opening {root_path}")
        with uproot.open(root_path) as f:
            tree = f["Events"]

            # sc8
            sc8_raw = self._get_branch(tree, "sc8PuppiEmuJets_")
            sc8 = self._transform_jets(sc8_raw)
            sc8 = ak.with_field(sc8, np.sqrt(sc8.mass), where="mass")

            # puppi
            puppi_raw = self._get_branch(tree, "puppiCands_")
            puppi = puppi_raw  # Or do any transform you want

        # Save each as Parquet
        sc8_name = f"{key}_sc8Emu_{file_index}.pq"
        puppi_name = f"{key}_puppi_{file_index}.pq"
        self._save_parquet(sc8, sc8_name)
        self._save_parquet(puppi, puppi_name)

    def _save_parquet(self, ak_arr, out_name) -> None:
        """
        Saves an Awkward array to Parquet in self.data_dir.
        """
        path = os.path.join(self.raw_path, out_name)
        ak.to_parquet(ak_arr, path)

    # -------------------------------------------------------------
    # Step 1: Get branch from ROOT
    # -------------------------------------------------------------
    def _get_branch(self, root_data, branch_str) -> ak.Array:
        """
        Identify all branches that match `branch_str` (excluding e.g. n{branch_str})
        and build an Awkward array with them.
        """
        branches = [
            b for b in root_data.keys()
            if branch_str in b and not f"n{branch_str}" in b
        ]
        data_arr = root_data.arrays(branches)
        # Example: "sc8PuppiEmuJets_pt" => rename it to "pt"
        renamed = {field: field.split('_', maxsplit=1)[-1] for field in branches}
        # Combine them into a record array; optionally name it "Momentum4D"
        arr = ak.with_name(
            ak.zip({renamed[k]: data_arr[k] for k in data_arr.fields}),
            "Momentum4D"
        )
        return arr
    
    def _transform_jets(self, jets_ak) -> ak.Array:
        # Keep only first jet
        jets = jets_ak[:, 0]

        # Gather daughter info
        arr_pt, arr_eta, arr_phi, arr_pdg, arr_vz = [], [], [], [], []
        arr_pt_rel, arr_log_pt = [], []
        arr_deta, arr_dphi = [], []
        arr_isfilled = []
        for i in range(self.nconst):
            arr_pt.append(jets[f"dau{i}_pt"])
            arr_eta.append(jets[f"dau{i}_eta"])
            arr_phi.append(jets[f"dau{i}_phi"])
            arr_pdg.append(jets[f"dau{i}_pdgId"])
            arr_vz.append(jets[f"dau{i}_vz"])

            arr_pt_rel.append(jets[f"dau{i}_pt"]/jets.pt)
            arr_log_pt.append(np.log(jets[f"dau{i}_pt"]))

            arr_deta.append(jets[f"dau{i}_eta"] - jets.eta)

            raw_dphi = jets[f"dau{i}_phi"] - jets.phi
            wrapped_dphi = (raw_dphi + np.pi) % (2 * np.pi) - np.pi
            arr_dphi.append(wrapped_dphi)
            
            pt_i = jets[f"dau{i}_pt"]
            arr_isfilled.append(ak.where(pt_i >= 0, 1, 0))

        constits = ak.with_name(
            ak.zip({
                "pt":  ak.zip(arr_pt),
                "eta": ak.zip(arr_eta),
                "phi": ak.zip(arr_phi),
                "pdg": ak.zip(arr_pdg),
                "vz":  ak.zip(arr_vz),
                "pt_rel": ak.zip(arr_pt_rel),
                "log_pt": ak.zip(arr_log_pt),
                "deta": ak.zip(arr_deta),
                "dphi": ak.zip(arr_dphi),
                "isfilled": ak.zip(arr_isfilled)
                
            }),
            "Momentum3D"
        )
        jets_with_constits = ak.with_name(
            ak.zip({
                "pt":      jets.pt,
                "eta":     jets.eta,
                "phi":     jets.phi,
                "mass":    jets.mass,
                "genpt":   jets.genpt,
                "gendr":   jets.gendr,
                "genmass": jets.genmass,
                "constituents": constits
            }),
            "Momentum4D"
        )
        return jets_with_constits

    # -------------------------------------------------------------
    # Step 2: Load Parquet
    # -------------------------------------------------------------
    def _load_parquet_files(self, prefix: str) -> ak.Array:
        """
        Loads all Parquet files in self.root that start with 'prefix' and
        concatenates them into a single Awkward array.
        """
        from os import listdir
        pq_files = sorted(
            f for f in listdir(self.raw_path)
            if f.startswith(prefix) and f.endswith(".pq")
        )
        arrays = []
        for pqf in pq_files:
            path = os.path.join(self.raw_path, pqf)
            print(f"  Loading {path}")
            arr = ak.from_parquet(path)
            arrays.append(arr)
    
        if len(arrays) == 0:
            print(f"  No Parquet files found with prefix '{prefix}'.")
            return None
    
        return ak.concatenate(arrays, axis=0)
    
    
    def _load_parquet(self) -> None:
        """
        Step 2: Load sc8/puppi from Parquet into Awkward arrays.
        """
        print("[_load_parquet] Loading sc8/puppi from Parquet...")
    
        # For background sc8
        self.sc8_bkg_ak = self._load_parquet_files("bkg_sc8Emu_")
        # For puppi background
        self.puppi_bkg_ak = self._load_parquet_files("bkg_puppi_")
    
        # For signal sc8
        self.sc8_sig_ak = self._load_parquet_files("sig_sc8Emu_")
        # For puppi signal
        self.puppi_sig_ak = self._load_parquet_files("sig_puppi_")
    
        # Optional: print quick summary
        print("  sc8_bkg:", type(self.sc8_bkg_ak), " length:", None if self.sc8_bkg_ak is None else len(self.sc8_bkg_ak))
        print("  puppi_bkg:", type(self.puppi_bkg_ak), " length:", None if self.puppi_bkg_ak is None else len(self.puppi_bkg_ak))
        print("  sc8_sig:", type(self.sc8_sig_ak), " length:", None if self.sc8_sig_ak is None else len(self.sc8_sig_ak))
        print("  puppi_sig:", type(self.puppi_sig_ak), " length:", None if self.puppi_sig_ak is None else len(self.puppi_sig_ak))
    # -------------------------------------------------------------
    # Step 3: Match Puppi
    # -------------------------------------------------------------
    def _match_puppi(self) -> None:
        """
        Step 3: Match Puppi to sc8 by pt, eta, phi (plus pdg if you like).
        """
        print("[_match_puppi] Matching sc8 to Puppi...")
        self.sc8_bkg_ak = self._match_puppi_weights(self.sc8_bkg_ak, self.puppi_bkg_ak)
        self.sc8_sig_ak = self._match_puppi_weights(self.sc8_sig_ak, self.puppi_sig_ak)

    def _match_puppi_weights(self, sc8_ak, puppi_ak) -> ak.Array:
        """
        Loop over events, call _match_single_event, return new Awkward array.
        """
        if sc8_ak is None or puppi_ak is None:
            raise RuntimeError("sc8_ak or puppi_ak is None!")
        if len(sc8_ak) != len(puppi_ak):
            raise ValueError("Need the same # of events in sc8 and puppi arrays.")

        from tqdm import tqdm
        updated = []
        for ev_sc8, ev_puppi in tqdm(zip(sc8_ak, puppi_ak),
                                     total=len(sc8_ak),
                                     desc="Matching Puppi to sc8"):
            updated.append(self._match_single_event(ev_sc8, ev_puppi))
        return ak.Array(updated)

    def _match_single_event(self, ev_sc8, ev_puppi) -> dict:
        """
        Fast per-event matching via dict lookup.  Includes your pt_rel & log_pt.
        """
        # build a lookup: (pt,eta,phi) -> (mass, puppiWeight)
        lookup = {
            (float(c["pt"]), float(c["eta"]), float(c["phi"])): 
            (float(c["mass"]), float(c["puppiWeight"]))
            for c in ev_puppi
        }

        matched_mass   = []
        matched_weight = []
        const = ev_sc8["constituents"]

        for j in range(self.nconst):
            pt_j   = float(const["pt"][str(j)])
            eta_j  = float(const["eta"][str(j)])
            phi_j  = float(const["phi"][str(j)])

            if pt_j < 0:
                matched_mass.append(-1.)
                matched_weight.append(-1.)
            else:
                m, w = lookup.get((pt_j, eta_j, phi_j), (0., 0.))
                matched_mass.append(m)
                matched_weight.append(w)

        # re‑assemble the constituents dict, including new fields
        constits_dict = {
            "pt":       const["pt"],
            "eta":      const["eta"],
            "phi":      const["phi"],
            "pdg":      const["pdg"],
            "vz":       const["vz"],
            "pt_rel":   const["pt_rel"],
            "log_pt":   const["log_pt"],
            "deta":     const["deta"],
            "dphi":     const["dphi"],
            "isfilled": const["isfilled"],
            "mass":     {str(i): matched_mass[i]   for i in range(self.nconst)},
            "puppiWeight": {str(i): matched_weight[i] for i in range(self.nconst)},
        }

        # top‑level event fields
        base = {
            "pt":   ev_sc8["pt"],
            "eta":  ev_sc8["eta"],
            "phi":  ev_sc8["phi"],
            "mass": ev_sc8["mass"],
        }
        for f in ("genpt","gendr","genmass"):
            if f in ev_sc8.fields:
                base[f] = ev_sc8[f]

        base["constituents"] = constits_dict
        return base


    # -------------------------------------------------------------
    # Step 4: One-hot PDG
    # -------------------------------------------------------------
    def _encode_pdg(self) -> None:
        """
        Wraps your existing one-hot creation for both bkg and sig.
        """
        print("[_encode_pdg] Performing PDG one-hot encoding...")
        self.sc8_bkg_ak = self._encode_pdg_func(self.sc8_bkg_ak)
        self.sc8_sig_ak = self._encode_pdg_func(self.sc8_sig_ak)
    
    def _encode_pdg_func(self, sc8_ak) -> ak.Array:
        """
        For each event, for each daughter j in [0..(n_constits-1)],
        and each PDG in self.pdg_order, create a record-of-32
        (like 'pdg_22': { '0': 1/0, '1': 1/0, ... }).
        Attach it to sc8_ak['constituents'] as new fields.
        """
        import awkward as ak
        from tqdm import tqdm
    
        if sc8_ak is None:
            return None
    
        updated_events = []
        for ev in tqdm(sc8_ak, total=len(sc8_ak), desc="One-Hot PDG encoding", leave=True):
            updated_events.append(self._encode_pdg_single_event(ev))
    
        encoded_ak = ak.Array(updated_events)
    
        # Optional debug prints
        if len(encoded_ak) > 0:
            self._debug_print_pdg_fields(encoded_ak)
    
        return encoded_ak
    
    def _encode_pdg_single_event(self, ev) -> dict:
        """
        Handle the one-hot PDG for a single event.
        """
        old_constits = ev["constituents"]
    
        # Copy existing fields
        new_constits = {
            "pt":  old_constits["pt"],
            #"eta": old_constits["eta"],
            #"phi": old_constits["phi"],
            "vz":  old_constits["vz"],
            "pt_rel": old_constits["pt_rel"],
            "log_pt": old_constits["log_pt"],
            "deta": old_constits["deta"],
            "dphi": old_constits["dphi"],
            "isfilled": old_constits["isfilled"]
        }
        # Copy mass, puppiWeight if present
        for extra_feat in ["mass", "puppiWeight"]:
            if extra_feat in old_constits.fields:
                new_constits[extra_feat] = old_constits[extra_feat]
    
        # Build the new fields for each PDG code
        for pdg_val in self.pdg_order:
            field_name = f"pdg_{pdg_val}"
            subdict = {}
            for j in range(self.nconst):
                pdg_j = old_constits["pdg"][str(j)]
                subdict[str(j)] = 1.0 if pdg_j == pdg_val else 0.0
            new_constits[field_name] = subdict
    
        # Build updated top-level event
        updated_event = {
            "pt":   ev["pt"],
            "eta":  ev["eta"],
            "phi":  ev["phi"],
            "mass": ev["mass"]
        }
        for gen_feat in ["genpt", "gendr", "genmass"]:
            if gen_feat in ev.fields:
                updated_event[gen_feat] = ev[gen_feat]
    
        updated_event["constituents"] = new_constits
        return updated_event
    
    def _debug_print_pdg_fields(self, encoded_ak) -> None:
        """
        Prints the fields in the first event's constituents and a few
        example values for PDG one-hot fields.
        """
        print("\n[DEBUG] After PDG one-hot encoding, the first event's constituent fields:")
        first_ev_constits = encoded_ak[0]["constituents"]
        print("Fields:", first_ev_constits.fields)
    
        one_hot_fields = [f for f in first_ev_constits.fields if f.startswith("pdg_")]
        for f_ in one_hot_fields[:3]:  # just example for the first few PDG fields
            vals_0to3 = [first_ev_constits[f_][str(j)] for j in range(4)]
            print(f"  {f_}, daughters 0..3 => {vals_0to3}")

    # -------------------------------------------------------------
    # Step 5: Pad => 3D arrays
    # -------------------------------------------------------------
    def _pad_to_3d(self) -> None:
        """
        Step 5: Convert sc8 bkg/sig Awkward arrays to 3D NumPy arrays and save them.
        """
        print("[_pad_to_3d] Converting data to 3D arrays...")
    
        # Define the features you want to extract
        feature_list = [
            "pt", 
            #"eta", "phi", 
            "vz", "pt_rel", "log_pt", "deta", "dphi", "isfilled", "mass", "puppiWeight"
        ] + [f"pdg_{p}" for p in self.pdg_order]
    
        # Convert background
        if self.sc8_bkg_ak is not None:
            self.bkg_3d = self._to_3d_numpy(self.sc8_bkg_ak, feature_list)
            np.save(os.path.join(self.preprocessed_path, f"bkg_3d_{self.preproc_output_name}.npy"), self.bkg_3d)
        else:
            self.bkg_3d = None
    
        # Convert signal
        if self.sc8_sig_ak is not None:
            self.sig_3d = self._to_3d_numpy(self.sc8_sig_ak, feature_list)
            np.save(os.path.join(self.preprocessed_path, f"sig_3d_{self.preproc_output_name}.npy"), self.sig_3d)
        else:
            self.sig_3d = None
    
    def _to_3d_numpy(self, sc8_ak, feature_list) -> np.ndarray:
        """
        Convert a 'record-of-nconst' sc8 Awkward array into a 3D NumPy array.
        Shape = (nEvents, n_constits, len(feature_list)).
        """
        import numpy as np
    
        nEv = len(sc8_ak)
        nF = len(feature_list)
    
        out = np.zeros((nEv, self.nconst, nF), dtype=np.float32)
    
        for i, event in enumerate(sc8_ak):
            constits = event["constituents"]
    
            for f_idx, feat in enumerate(feature_list):
                if feat not in constits.fields:
                    out[i, :, f_idx] = -999.0
                    continue
    
                feat_record = constits[feat]
                for j in range(self.nconst):
                    key_j = str(j)
                    out[i, j, f_idx] = feat_record[key_j] if key_j in feat_record.fields else -999.0
    
        return out
    
    def load_3d_data(self) -> None:
        """
        Optional method to reload your saved 3D arrays from disk.
        """
        bkg_path = os.path.join(self.preprocessed_path, f"bkg_3d_{self.preproc_output_name}.npy")
        sig_path = os.path.join(self.preprocessed_path, f"sig_3d_{self.preproc_output_name}.npy")
    
        if os.path.isfile(bkg_path):
            self.bkg_3d = np.load(bkg_path)
            print(f"Loaded bkg_3d from {bkg_path} with shape {self.bkg_3d.shape}")
    
        if os.path.isfile(sig_path):
            self.sig_3d = np.load(sig_path)
            print(f"Loaded sig_3d from {sig_path} with shape {self.sig_3d.shape}")

    # -------------------------------------------------------------
    # Step 6: Combine bkg & sig => X, y
    # -------------------------------------------------------------
    def _combine_sig_bkg(self) -> None:
        """
        Step 6: Combine bkg & sig arrays => self.x, self.y.
        """
        print("[_combine_sig_bkg] Combining bkg & sig => X, y...")
    
        if (self.bkg_3d is None) or (self.sig_3d is None):
            print("  Missing bkg or sig 3D arrays. Skipping.")
            self.x = None
            self.y = None
            return

        # 1) Combine bkg/sig
        X = np.concatenate([self.bkg_3d, self.sig_3d], axis=0)

        # 2) integer labels
        labels = np.concatenate([
            np.zeros(len(self.bkg_3d), dtype=np.int32),
            np.ones(len(self.sig_3d), dtype=np.int32)
        ])

        # 3) convert to one-hot (n_classes=2 here)
        n_classes = 2
        y = np.eye(n_classes, dtype=np.float32)[labels]
    
        self.x = X
        self.y = y
        print("Combined X shape:", X.shape, " y shape:", y.shape)

    # -------------------------------------------------------------
    # Step 7: Normalize (optional)
    # -------------------------------------------------------------
    def _normalize_data(self) -> None:
        """
        Step 7: Normalize data and optionally split into train/test with plots.
        """
        print("[_normalize_data] Normalizing data...")
    
        # No data to normalize
        if self.x is None or self.y is None:
            print("  No data found. Skipping normalization.")
            return
    
        # Choose the same feature list used to build self.x
        feature_list = [
            "pt", 
            #"eta", "phi", 
            "vz", "pt_rel", "log_pt", "deta", "dphi", "isfilled", "mass", "puppiWeight"
        ] + [f"pdg_{p}" for p in self.pdg_order]
    
        # Perform normalization and splitting
        self.x_train, self.x_test, self.y_train, self.y_test, self.scaler = self._norm_split(
            self.x, self.y, features=feature_list
        )

        # We store both the training and test data in the processed folder
        # for later use
     
        print("Setting x.self, y.self to training split")
        self.x = self.x_train
        self.y = self.y_train
        #save the normalized data
        np.save(os.path.join(self.processed, f"proc_train_{self.preproc_output_name}.npy"), self.x)
        np.save(os.path.join(self.processed, f"proc_labels_train_{self.preproc_output_name}.npy"), self.y)

        #save the normalized data
        np.save(os.path.join(self.processed, f"proc_test_{self.preproc_output_name}.npy"), self.x_test)
        np.save(os.path.join(self.processed, f"proc_labels_test_{self.preproc_output_name}.npy"), self.y_test)
    
        print("After normalization, shapes:")
        print("  self.x:", self.x.shape, " self.y:", self.y.shape)

    def _plot_all_features(self, X_3d, features, stage="before") -> None:
        """
        Plots a histogram for each feature in features from the 3D array,
        using only the values for daughter 0 across all events.
        X_3d shape: (numEvents, numConstituents, numFeatures).
        """
        num_events, num_constits, num_feats = X_3d.shape

        for i, feat_name in enumerate(features):
            # Only take the 0th daughter across all events
            feat_values = X_3d[:, 0, i]

            plt.figure()
            plt.hist(feat_values, bins=50, histtype="step", label=f"{feat_name} ({stage})")
            plt.xlabel(feat_name)
            #do a log scale for the y axis
            plt.yscale("log")
            plt.grid()
            plt.ylabel("Count")
            plt.title(f"Distribution of {feat_name} ({stage} normalization) for daughter 0")
            plt.legend()
            plt.show()
            #save the plot in the preprocessed folder if stage is before if it is after then save in processed
            if stage == "before":
                plt.savefig(os.path.join(self.preprocessed_path, f"{feat_name}_{stage}.png"))
            else:
                plt.savefig(os.path.join(self.processed, f"{feat_name}_{stage}.png"))
            plt.close()
        print(f"Plotted all features for {stage} normalization.")
            
    def load_normalized_data(self) -> None:
        """
        Optional method to reload your saved normalized data from disk.
        """
        train_path = os.path.join(self.processed, f"train_{self.proc_output_name}.npy")
        test_path = os.path.join(self.processed, f"test_{self.proc_output_name}.npy")

        if os.path.isfile(train_path):
            self.x = np.load(train_path)
            print(f"Loaded x_train from {train_path} with shape {self.x.shape}")

        if os.path.isfile(test_path):
            self.x = np.load(test_path)
            print(f"Loaded x_test from {test_path} with shape {self.x.shape}")
        # Load labels   
        train_labels_path = os.path.join(self.processed, f"train_labels_{self.proc_output_name}.npy")
        test_labels_path = os.path.join(self.processed, f"test_labels_{self.proc_output_name}.npy")

        if os.path.isfile(train_labels_path):
            self.y = np.load(train_labels_path)
            print(f"Loaded y_train from {train_labels_path} with shape {self.y.shape}")
        if os.path.isfile(test_labels_path):
            self.y = np.load(test_labels_path)
            print(f"Loaded y_test from {test_labels_path} with shape {self.y.shape}")

    # ---------------------------------------------------------------------
    # I) Normalize, Split
    # ---------------------------------------------------------------------
    def _norm_split(self, X, y, features) -> tuple:
        """
        Flatten X to 2D => scale => train/test => reshape back if desired.
        Return X_train, X_test, y_train, y_test, scaler.
        """

        # 1) Flatten and Mask the missing constituents with nan so that they are ignored in the scaling
        nEv, nC, nF = X.shape
        # We need to check if the pt value is -1 and if yes we set all features to np.nan
        mask = (X[:, :, 0:1] == -1)               
        X = X.astype(np.float32)                   
        X = np.where(mask, np.nan, X)  

        # --- DEBUG #1: puppiWeight summary BEFORE split ---
        pw_idx = features.index("puppiWeight")
        pw_all = X[:, :, pw_idx]
        print(f"\n[DEBUG] puppiWeight (daughter 0) before split:")
        print("  Sample values:", pw_all[:3, 0])            # first 3 events
        print("  Min:", np.nanmin(pw_all), " Max:", np.nanmax(pw_all))
        print("  Mean:", np.nanmean(pw_all))

        #mask = X == -1
        #np.where(np.sum(mask, axis=1) < mask.shape[-1], True, False)
        #X[mask] = np.nan
        X2d = X.reshape(nEv, nC * nF)

        # 2) Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X2d, y, test_size=self.test_size, random_state=self.random_state
        )

        # --- PLOT BEFORE SCALING ON TRAINING SUBSET ---
        print("\nPlotting all features before scaling (training set)...")
        X_train_3d_before = X_train.reshape(-1, nC, nF)
        self._plot_all_features(X_train_3d_before, features, stage="before")

        # --- DEBUG #2: puppiWeight summary AFTER split, BEFORE scale ---
        # reshape back just for the debug
        tr_pw = X_train.reshape(-1, nC, nF)[:, :, pw_idx]
        te_pw = X_test .reshape(-1, nC, nF)[:, :, pw_idx]
        print(f"\n[DEBUG] puppiWeight after split (no scaling):")
        print("  TRAIN Sample:", tr_pw[:3, 0])
        print("    Min:", np.nanmin(tr_pw), " Max:", np.nanmax(tr_pw), " Mean:", np.nanmean(tr_pw))
        print("  TEST  Sample:", te_pw[:3, 0])
        print("    Min:", np.nanmin(te_pw), " Max:", np.nanmax(te_pw), " Mean:", np.nanmean(te_pw))


        # 3) Figure out which features to normalize
        skip = {"pt_rel"} | {"isfilled"} | {"puppiWeight"} | {f"pdg_{p}" for p in self.pdg_order}
        feat_to_norm = [feat for feat in features if feat not in skip]
        feat_idx     = [features.index(feat) for feat in feat_to_norm]
    

        # 4) Build column indices across all constituents
        col_idxs = sorted(
            c * nF + f
            for c in range(nC)
            for f in feat_idx
        )
        print("\nColumn indices to normalize:", col_idxs[:8], "... (total", len(col_idxs), ")")

        # 5) Scale only those columns
        scaler = RobustScaler()
        X_train[:, col_idxs] = scaler.fit_transform(X_train[:, col_idxs])
        X_test[:,  col_idxs] = scaler.transform(X_test[:,  col_idxs])

        # --- DEBUG #3: puppiWeight summary AFTER scaling (should be unchanged) ---
        tr_pw_after = X_train.reshape(-1, nC, nF)[:, :, pw_idx]
        te_pw_after = X_test .reshape(-1, nC, nF)[:, :, pw_idx]
        print(f"\n[DEBUG] puppiWeight after scaling (d0):")
        print("  TRAIN Sample:", tr_pw_after[:3, 0])
        print("    Min:", np.nanmin(tr_pw_after), " Max:", np.nanmax(tr_pw_after), " Mean:", np.nanmean(tr_pw_after))
        print("  TEST  Sample:", te_pw_after[:3, 0])
        print("    Min:", np.nanmin(te_pw_after), " Max:", np.nanmax(te_pw_after), " Mean:", np.nanmean(te_pw_after))



        # 6) reshape back to 3D
        # X_train.shape = (nEv, nC*nF) => (nEv, nC, nF)
        # X_test.shape  = (nEv, nC*nF) => (nEv, nC, nF)
        X_train_3d = X_train.reshape(-1, nC, nF)
        X_test_3d  = X_test. reshape(-1, nC, nF)

        # 7) Check for NaN values and replace them with self.padding_value
        X_train_3d = np.nan_to_num(X_train_3d, nan=self.padding_value)
        X_test_3d  = np.nan_to_num(X_test_3d, nan=self.padding_value)

         # Plot after scaling (on the training subset, for example)
        print("Plotting all features after scaling...")
        self._plot_all_features(X_train_3d, features, stage="after")

        return X_train_3d, X_test_3d, y_train, y_test, scaler

    # -------------------------------------------------------------
    # Step 7b: K-fold splitting
    # -------------------------------------------------------------
    def _split_kfold(self) -> None:
        """Split into k folds if requested."""
        if self.kfolds > 0 and self.train:
            print(f"[_split_kfold] Using {self.kfolds}-fold cross-validation...")
            skf = sklearn.model_selection.StratifiedKFold(
                n_splits=self.kfolds, shuffle=True, random_state=self.seed
            )
            # We have self.x (shape [n_samples, …]) and self.y (labels [n_samples])
            # Build the list of (train_idx, valid_idx) pairs:
            #self.kfolds = list(skf.split(self.x, self.y))

            # Convert back from one-hot to class targets since sklearn function does not
            # like one-hot targets.
            self.kfolds = skf.split(self.x, np.argmax(self.y, axis=-1))

            # Possibly store the splits for usage later.

    # -------------------------------------------------------------
    # Step 8: Load test data
    # -------------------------------------------------------------
    def load_proc_test_data(self) -> None:
        """
        Load the test data from the processed folder.
        """
        test_path = os.path.join(self.processed, f"proc_test_{self.preproc_output_name}.npy")
        test_labels_path = os.path.join(self.processed, f"proc_labels_test_{self.preproc_output_name}.npy")

        if os.path.isfile(test_path):
            self.x = np.load(test_path)
            print(f"Loaded x_test from {test_path} with shape {self.x.shape}")

        if os.path.isfile(test_labels_path):
            self.y = np.load(test_labels_path)
            print(f"Loaded y_test from {test_labels_path} with shape {self.y.shape}")

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------
    def show_details(self) -> None:
        """Print some info about current data shapes."""
        print("Data shape X:", None if self.x is None else self.x.shape)
        print("Data shape y:", None if self.y is None else self.y.shape)
        print("Number of folds:", self.kfolds)



