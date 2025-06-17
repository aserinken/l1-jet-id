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
    """Data Class for importing and processing top pt jets.

    Raw data is available in /eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/lroberts/
    But the necessary raw files were uploaded to a local folder.
    If one has access to the EOS folder, one can set the path to the EOS folders
    As the data format is different to the SC4 data, one has to load different branches and do a matching

    Args:
        root: The root directory of the data. It should contain a 'number of constituents' folder with
            'raw' and 'preprocessed' folders inside. If the folders does not exist, they will be created.
        nconst: Number of constituents to pad/truncate to.
        pdg_order: PDG codes to one-hot encode. By default, it is set to [22, -11, 11, -13, 13, 130, -211, 211] which is the same order as in the SC4 algorithm.
        norm: Chosen normalization.
        train: Whether this is training data or validation/test.
        kfolds: Number of splits for k-fold cross-validation.
        seed: Random seed for reproducibility.
        datasets: Dictionary with keys ("bkg", "sig") and values = list of ROOT paths.
        test_size: Fraction for train/test split.
        random_state: Random state for train/test split.
        padding_value: Value to use for padding the data. Default it is set to 0.
            The original value that is used for missing values inside the ROOT files is -1.
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
        datasets: dict = None,
        test_size: float = 0.2,
        random_state: int = 42,
        padding_value: float = 0,
        analysis: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.nconst = nconst
        self.pdg_order = (
            pdg_order if pdg_order else [22, -11, 11, -13, 13, 130, -211, 211]
        )
        self.norm = norm
        self.train = train
        self.kfolds = kfolds
        self.seed = seed
        self.datasets = datasets if datasets else {}
        self.test_size = test_size
        self.random_state = random_state
        self.padding_value = padding_value
        self.analysis = analysis

        # ---------------------------------------------------------------------
        # A) Define extra attributes
        # ---------------------------------------------------------------------
        # Paths for top level folder
        self.top_level = self.root / f"{nconst}const"

        # Paths for raw, preprocessed and processed folders inside the top level folder
        self.type = "train" if train else "val"
        self.raw_path = self.top_level / "raw"
        self.preprocessed_path = self.top_level / "preprocessed"
        self.processed_path = self.top_level / "processed"

        # Names for the output files inside the preprocessed and processed folders
        self.preproc_output_name = f"{self.nconst}const"
        self.proc_output_name = f"{self.type}_{self.norm}_{self.nconst}const"

        # Final data
        self.x = None
        self.y = None

        # Internal placeholders
        self.sc8_bkg_ak = None
        self.sc8_sig_ak = None
        self.puppi_bkg_ak = None
        self.puppi_sig_ak = None

        # ---------------------------------------------------------------------
        # B) Check if structure of the folders is correct, if not create them
        # ---------------------------------------------------------------------
        self._check_create_top_level_folder()
        self._check_create_folders()

        # ---------------------------------------------------------------------
        # C) Run the pipeline
        # ---------------------------------------------------------------------
        # Check if the data is for training or validation/test
        if self.train == False:
            print("Loading data for validation/test from processed folder.")
            self.load_proc_test_data()
            self.njets = self.x.shape[0]
            self.nfeats = self.x.shape[-1]
            return


        # Check if the preprocessed data exists and load it, otherwise run the full pipeline
        if self._check_preprocessed_data_exists() == True:
            print("Preprocessed data found. Loading from 'preprocessed' folder.")
            # Check if the processed data exists and load it, otherwise run the pipeline with the preprocessed data
            if self._check_processed_data_exists() == True:
                print("Processed data found. Loading from 'processed' folder.")
                # 1) Load the processed data
                self.load_normalized_data()
                # 2) K-fold splitting
                self._split_kfold()
            else:
                print("Processed data not found. Loading from 'preprocessed' folder.")
                # 1) Load the preprocessed data
                self.load_3d_data()
                # 2) Combine bkg/sig => X, y
                self._combine_sig_bkg()
                # 3) Normalize or split
                self._normalize_data()
                # 4) K-fold splitting
                self._split_kfold()
        else:
            print("No preprocssed data found. Running full pipeline.")
            # 1) Convert ROOT -> Parquet
            self._process_root_files()
            # 2) Load sc8/puppi from Parquet
            self._load_parquet()
            # 3) Match sc8 to Puppi
            self._match_puppi()
            # 4) One-hot PDG
            self._encode_pdg()
            # 5) Akward -> 3D numpy and save
            self._pad_to_3d()
            # 6) Load the 3D data
            self.load_3d_data()
            # 7) Combine bkg/sig => X, y
            self._combine_sig_bkg()
            # 8) Normalize or split
            self._normalize_data()
            # 9) K-fold splitting
            self._split_kfold()

        # Set the number of jets and features for later use
        self.njets = self.x.shape[0]
        self.nfeats = self.x.shape[-1]

    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # 1) Check if self.top_level folder exists, if not create it
    # ---------------------------------------------------------------------
    def _check_create_top_level_folder(self) -> None:
        """
        Check if the top-level folder named f"{self.nconst}const" exists in self.root, if not create it.
        """
        if not self.top_level.is_dir():
            print(f"Creating top-level folder: {self.top_level}")
            os.makedirs(self.top_level)
        else:
            print(f"Top-level folder already exists: {self.top_level}")

    # ---------------------------------------------------------------------
    # 2) Check if the folders "raw", "preprocessed", "processed" exist in self.top_level if not create them inside self.top_level
    # ---------------------------------------------------------------------
    def _check_create_folders(self) -> None:
        """
        Check if the folders "raw", "preprocessed", "processed" exist in self.top_level, if not create them inside self.top_level.
        """
        for folder in ["raw", "preprocessed", "processed"]:
            folder_path = self.top_level / folder
            if not folder_path.is_dir():
                print(f"Creating folder: {folder_path}")
                os.makedirs(folder_path)
            else:
                print(f"Folder already exists: {folder_path}")

    # ---------------------------------------------------------------------
    # 3) Check if the preprocessed data exists
    # ---------------------------------------------------------------------
    def _check_preprocessed_data_exists(self) -> bool:
        """
        Check if the preprocessed data exists in the preprocessed folder and if yes, return True.
        """
        needed_files = [
            f"bkg_3d_{self.preproc_output_name}.npy",
            f"sig_3d_{self.preproc_output_name}.npy",
        ]
        if not self.preprocessed_path.is_dir():
            return False
        return all(
            (self.preprocessed_path / file_name).is_file() for file_name in needed_files
        )

    # ---------------------------------------------------------------------
    # 4) Check if the processed data exists
    # ---------------------------------------------------------------------
    def _check_processed_data_exists(self) -> bool:
        """
        Check if the processed data exists in the processed folder and if yes, return True.
        """
        needed_files = [
            f"proc_{self.proc_output_name}.npy",
            f"proc_labels_{self.proc_output_name}.npy",
        ]
        if not self.processed_path.is_dir():
            return False
        return all(
            (self.processed_path / file_name).is_file() for file_name in needed_files
        )

    # ---------------------------------------------------------------------
    # 5) Process all ROOT files -> Parquet
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

        Args:
            key: The key for the dataset ("bkg" or "sig").
            file_list: List of ROOT files to process.
        """
        for file_index, root_path in enumerate(file_list):
            if not os.path.exists(root_path):
                print(f"  file not found: {root_path}, skipping.")
                continue
            self._process_single_file(key, file_index, root_path)

    def _process_single_file(self, key, file_index, root_path) -> None:
        """
        Convert one ROOT file (both sc8 and puppi (puppi is used for matching)) into Parquet and saved it.

        Args:
            key: The key for the dataset ("bkg" or "sig").
            file_index: Index of the file in the list.
            root_path: Path to the indexed ROOT file.
        """
        print(f"  Opening {root_path}")
        with uproot.open(root_path) as f:
            tree = f["Events"]

            sc8_raw = self._get_branch(tree, "sc8PuppiEmuJets_")
            sc8 = self._transform_jets(sc8_raw)
            sc8 = ak.with_field(sc8, np.sqrt(sc8.mass), where="mass")

            puppi = self._get_branch(tree, "puppiCands_")

        sc8_name = f"{key}_sc8Emu_{self.nconst}_{file_index}.pq"
        puppi_name = f"{key}_puppi_{self.nconst}_{file_index}.pq"
        self._save_parquet(sc8, sc8_name)
        self._save_parquet(puppi, puppi_name)

    def _save_parquet(self, ak_arr, out_name) -> None:
        """
        Saves an Awkward array to Parquet in self.data_dir.

        Args:
            ak_arr: The Awkward array to save.
            out_name: The name of the output file, e.g. "bkg_sc8Emu_32_0.pq".
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

        Args:
            root_data: The data from the ROOT file, either as an Awkward array or Uproot tree.
            branch_str: Name of the branch to look for, e.g. "sc8PuppiEmuJets_".
        """
        branches = [
            b for b in root_data.keys() if branch_str in b and not f"n{branch_str}" in b
        ]
        data_arr = root_data.arrays(branches)
        renamed = {field: field.split("_", maxsplit=1)[-1] for field in branches}
        arr = ak.with_name(
            ak.zip({renamed[k]: data_arr[k] for k in data_arr.fields}), "Momentum4D"
        )
        return arr

    def _transform_jets(self, jets_ak) -> ak.Array:
        """
        Transform the jets Awkward array to include only the first jet(being the top pt jet of the event) and its constituents.
        It gathers the daughter information and computes relative pt, log(pt), deta, dphi, and isfilled.

        Args:
            jets_ak: The Awkward array of jets, with fields like "dau0_pt", "dau0_eta", etc.
        """
        jets = jets_ak[:, 0]
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
            arr_pt_rel.append(jets[f"dau{i}_pt"] / jets.pt)
            arr_log_pt.append(np.log(jets[f"dau{i}_pt"]))
            arr_deta.append(jets[f"dau{i}_eta"] - jets.eta)

            raw_dphi = jets[f"dau{i}_phi"] - jets.phi
            wrapped_dphi = (raw_dphi + np.pi) % (2 * np.pi) - np.pi
            arr_dphi.append(wrapped_dphi)

            pt_i = jets[f"dau{i}_pt"]
            arr_isfilled.append(ak.where(pt_i >= 0, 1, 0))

        constits = ak.with_name(
            ak.zip(
                {
                    "pt": ak.zip(arr_pt),
                    "eta": ak.zip(arr_eta),
                    "phi": ak.zip(arr_phi),
                    "pdg": ak.zip(arr_pdg),
                    "vz": ak.zip(arr_vz),
                    "pt_rel": ak.zip(arr_pt_rel),
                    "log_pt": ak.zip(arr_log_pt),
                    "deta": ak.zip(arr_deta),
                    "dphi": ak.zip(arr_dphi),
                    "isfilled": ak.zip(arr_isfilled),
                }
            ),
            "Momentum3D",
        )
        jets_with_constits = ak.with_name(
            ak.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "genpt": jets.genpt,
                    "gendr": jets.gendr,
                    "genmass": jets.genmass,
                    "constituents": constits,
                }
            ),
            "Momentum4D",
        )
        return jets_with_constits

    # -------------------------------------------------------------
    # Step 2: Load Parquet
    # -------------------------------------------------------------
    def _load_parquet_files(self, prefix: str) -> ak.Array:
        """
        Loads all Parquet files in self.root that start with 'prefix' and
        concatenates them into a single Awkward array.

        Args:
            prefix: The prefix to filter Parquet files, e.g. "bkg_sc8Emu_32_".
        """
        from os import listdir

        pq_files = sorted(
            f
            for f in listdir(self.raw_path)
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
        Loads the sc8 and Puppi data from Parquet files in the raw_path directory.
        """
        print("[_load_parquet] Loading sc8/puppi from Parquet...")

        self.sc8_bkg_ak = self._load_parquet_files(f"bkg_sc8Emu_{self.nconst}_")
        self.puppi_bkg_ak = self._load_parquet_files(f"bkg_puppi_{self.nconst}_")

        self.sc8_sig_ak = self._load_parquet_files(f"sig_sc8Emu_{self.nconst}_")
        self.puppi_sig_ak = self._load_parquet_files(f"sig_puppi_{self.nconst}_")

        print(
            "  sc8_bkg:",
            type(self.sc8_bkg_ak),
            " length:",
            None if self.sc8_bkg_ak is None else len(self.sc8_bkg_ak),
        )
        print(
            "  puppi_bkg:",
            type(self.puppi_bkg_ak),
            " length:",
            None if self.puppi_bkg_ak is None else len(self.puppi_bkg_ak),
        )
        print(
            "  sc8_sig:",
            type(self.sc8_sig_ak),
            " length:",
            None if self.sc8_sig_ak is None else len(self.sc8_sig_ak),
        )
        print(
            "  puppi_sig:",
            type(self.puppi_sig_ak),
            " length:",
            None if self.puppi_sig_ak is None else len(self.puppi_sig_ak),
        )

    # -------------------------------------------------------------
    # Step 3: Match Puppi
    # -------------------------------------------------------------
    def _match_puppi(self) -> None:
        """
        Matches the sc8 jets to Puppi weights
        """
        print("[_match_puppi] Matching sc8 to Puppi...")
        self.sc8_bkg_ak = self._match_puppi_weights(self.sc8_bkg_ak, self.puppi_bkg_ak)
        self.sc8_sig_ak = self._match_puppi_weights(self.sc8_sig_ak, self.puppi_sig_ak)

    def _match_puppi_weights(self, sc8_ak, puppi_ak) -> ak.Array:
        """
        Loop over events, call _match_single_event, return new Awkward array.

        Args:
            sc8_ak: Awkward array of sc8 jets.
            puppi_ak: Awkward array of Puppi jets.
        """
        if sc8_ak is None or puppi_ak is None:
            raise RuntimeError("sc8_ak or puppi_ak is None!")
        if len(sc8_ak) != len(puppi_ak):
            raise ValueError("Need the same # of events in sc8 and puppi arrays.")

        updated = []
        for ev_sc8, ev_puppi in tqdm(
            zip(sc8_ak, puppi_ak), total=len(sc8_ak), desc="Matching Puppi to sc8"
        ):
            updated.append(self._match_single_event(ev_sc8, ev_puppi))
        return ak.Array(updated)

    def _match_single_event(self, ev_sc8, ev_puppi) -> dict:
        """
        Per-event matching via dict lookup. Pt, eta, phi of each sc8 constituent 
        is matched to the Puppi mass and weight. Then the constituents dict is updated
        with the matched mass and weight.

        Args:
            ev_sc8: Single event from sc8 Awkward array.
            ev_puppi: Single event from Puppi Awkward array.
        """
        lookup = {
            (float(c["pt"]), float(c["eta"]), float(c["phi"])): (
                float(c["mass"]),
                float(c["puppiWeight"]),
            )
            for c in ev_puppi
        }

        matched_mass = []
        matched_weight = []
        const = ev_sc8["constituents"]

        for j in range(self.nconst):
            pt_j = float(const["pt"][str(j)])
            eta_j = float(const["eta"][str(j)])
            phi_j = float(const["phi"][str(j)])

            if pt_j < 0:
                matched_mass.append(-1.0)
                matched_weight.append(-1.0)
            else:
                m, w = lookup.get((pt_j, eta_j, phi_j), (0.0, 0.0))
                matched_mass.append(m)
                matched_weight.append(w)

        constits_dict = {
            "pt": const["pt"],
            "eta": const["eta"],
            "phi": const["phi"],
            "pdg": const["pdg"],
            "vz": const["vz"],
            "pt_rel": const["pt_rel"],
            "log_pt": const["log_pt"],
            "deta": const["deta"],
            "dphi": const["dphi"],
            "isfilled": const["isfilled"],
            "mass": {str(i): matched_mass[i] for i in range(self.nconst)},
            "puppiWeight": {str(i): matched_weight[i] for i in range(self.nconst)},
        }

        base = {
            "pt": ev_sc8["pt"],
            "eta": ev_sc8["eta"],
            "phi": ev_sc8["phi"],
            "mass": ev_sc8["mass"],
        }
        for f in ("genpt", "gendr", "genmass"):
            if f in ev_sc8.fields:
                base[f] = ev_sc8[f]

        base["constituents"] = constits_dict
        return base

    # -------------------------------------------------------------
    # Step 4: One-hot PDG
    # -------------------------------------------------------------
    def _encode_pdg(self) -> None:
        """
        Encodes the PDG codes of the constituents in sc8_bkg_ak and sc8_sig_ak
        """
        print("[_encode_pdg] Performing PDG one-hot encoding...")
        self.sc8_bkg_ak = self._encode_pdg_func(self.sc8_bkg_ak)
        self.sc8_sig_ak = self._encode_pdg_func(self.sc8_sig_ak)

    def _encode_pdg_func(self, sc8_ak) -> ak.Array:
        """
        For each event, for each daughter j in [0..(n_const - 1)],
        and each PDG in self.pdg_order, create a record-of-32
        (like 'pdg_22': { '0': 1/0, '1': 1/0, ... }).
        Attach it to sc8_ak['constituents'] as new fields.

        Args:
            sc8_ak: Awkward array of sc8 jets, with fields like "constituents".
        """
        import awkward as ak
        from tqdm import tqdm

        if sc8_ak is None:
            return None

        updated_events = []
        for ev in tqdm(
            sc8_ak, total=len(sc8_ak), desc="One-Hot PDG encoding", leave=True
        ):
            updated_events.append(self._encode_pdg_single_event(ev))

        encoded_ak = ak.Array(updated_events)
        return encoded_ak

    def _encode_pdg_single_event(self, ev) -> dict:
        """
        Handle the one-hot PDG for a single event. 
        It creates a new 'constituents' field with one-hot encoded PDG codes
        and copies existing fields from the original event. One can also 
        decide here which features to keep or drop. As in the SC4 algorithm,
        we keep pt, vz, pt_rel, log_pt, deta, dphi, isfilled, and mass/puppiWeight
        from the original constituents. 

        Args:
            ev: Single event from the sc8 Awkward array, with fields like "constituents".
        """
        old_constits = ev["constituents"]
        new_constits = {
            "pt": old_constits["pt"],
            # "eta": old_constits["eta"],
            # "phi": old_constits["phi"],
            "vz": old_constits["vz"],
            "pt_rel": old_constits["pt_rel"],
            "log_pt": old_constits["log_pt"],
            "deta": old_constits["deta"],
            "dphi": old_constits["dphi"],
            "isfilled": old_constits["isfilled"],
        }

        for extra_feat in ["mass", "puppiWeight"]:
            if extra_feat in old_constits.fields:
                new_constits[extra_feat] = old_constits[extra_feat]

        for pdg_val in self.pdg_order:
            field_name = f"pdg_{pdg_val}"
            subdict = {}
            for j in range(self.nconst):
                pdg_j = old_constits["pdg"][str(j)]
                subdict[str(j)] = 1.0 if pdg_j == pdg_val else 0.0
            new_constits[field_name] = subdict

        updated_event = {
            "pt": ev["pt"],
            "eta": ev["eta"],
            "phi": ev["phi"],
            "mass": ev["mass"],
        }
        for gen_feat in ["genpt", "gendr", "genmass"]:
            if gen_feat in ev.fields:
                updated_event[gen_feat] = ev[gen_feat]

        updated_event["constituents"] = new_constits
        return updated_event
    
    # -------------------------------------------------------------
    # Step 5: Pad => 3D arrays
    # -------------------------------------------------------------
    def _pad_to_3d(self) -> None:
        """
        Convert sc8 bkg/sig Awkward arrays to 3D NumPy arrays and save them.
        """
        print("[_pad_to_3d] Converting data to 3D arrays...")

        feature_list = [
            "pt",
            # "eta", "phi",
            "vz",
            "pt_rel",
            "log_pt",
            "deta",
            "dphi",
            "isfilled",
            "mass",
            "puppiWeight",
        ] + [f"pdg_{p}" for p in self.pdg_order]

        top_level_list = ["pt", "eta", "phi", "mass"]

        if self.sc8_bkg_ak is not None:
            self.bkg_3d = self._to_3d_numpy(self.sc8_bkg_ak, feature_list)
            np.save(
                os.path.join(
                    self.preprocessed_path, f"bkg_3d_{self.preproc_output_name}.npy"
                ),
                self.bkg_3d,
            )

            bkg_top = self._to_2d_numpy_toplevel(self.sc8_bkg_ak, top_level_list)
            np.save(
                os.path.join(
                    self.preprocessed_path, f"bkg_top_{self.preproc_output_name}.npy"
                ),
                bkg_top,
            )
        else:
            self.bkg_3d = None

        if self.sc8_sig_ak is not None:
            self.sig_3d = self._to_3d_numpy(self.sc8_sig_ak, feature_list)
            np.save(
                os.path.join(
                    self.preprocessed_path, f"sig_3d_{self.preproc_output_name}.npy"
                ),
                self.sig_3d,
            )

            sig_top = self._to_2d_numpy_toplevel(self.sc8_sig_ak, top_level_list)
            np.save(
                os.path.join(
                    self.preprocessed_path, f"sig_top_{self.preproc_output_name}.npy"
                ),
                sig_top,
            )
        else:
            self.sig_3d = None

    def _to_3d_numpy(self, sc8_ak, feature_list) -> np.ndarray:
        """
        Convert a 'record-of-nconst' sc8 Awkward array into a 3D NumPy array.
        Shape = (nEvents, n_constits, len(feature_list)). 
        Each feature is extracted from the 'constituents' field of the sc8_ak array.

        Args:
            sc8_ak: Awkward array of sc8 jets, with fields like "constituents".
            feature_list: List of features to extract from the constituents.
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
                    out[i, j, f_idx] = (
                        feat_record[key_j] if key_j in feat_record.fields else -999.0
                    )

        return out

    def _to_2d_numpy_toplevel(self, sc8_ak, top_level_list) -> np.ndarray:
        """
        Extracts the top-level features from the sc8_ak array and converts them to a 2D NumPy array.
        Shape = (nEvents, len(top_level_list)).
        Each feature is extracted from the top-level fields of the sc8_ak array.

        Args:
            sc8_ak: Awkward array of sc8 jets, with fields like "pt", "eta", "phi", "mass".
            top_level_list: List of top-level features to extract.
        """
        import numpy as np

        nEv = len(sc8_ak)
        nF = len(top_level_list)

        out = np.zeros((nEv, nF), dtype=np.float32)

        for i, event in enumerate(sc8_ak):
            for f_idx, feat in enumerate(top_level_list):
                if feat not in event.fields:
                    out[i, f_idx] = self.padding_value
                else:
                    out[i, f_idx] = (
                        event[feat] if event[feat] is not None else self.padding_value
                    )
        return out

    def load_3d_data(self) -> None:
        """
        Loads the 3D data from the preprocessed folder. 
        Prints the shapes of the loaded arrays.
        If the top-level 2D arrays are also available, they are loaded as well.
        If the 3D arrays are not found, it will not raise an error but will skip the loading.
        Note: This method assumes that the preprocessed data has been saved in the expected format.
        """
        bkg_path = os.path.join(
            self.preprocessed_path, f"bkg_3d_{self.preproc_output_name}.npy"
        )
        sig_path = os.path.join(
            self.preprocessed_path, f"sig_3d_{self.preproc_output_name}.npy"
        )

        bkg_top_path = os.path.join(
            self.preprocessed_path, f"bkg_top_{self.preproc_output_name}.npy"
        )
        sig_top_path = os.path.join(
            self.preprocessed_path, f"sig_top_{self.preproc_output_name}.npy"
        )

        if os.path.isfile(bkg_path):
            self.bkg_3d = np.load(bkg_path)
            print(f"Loaded bkg_3d from {bkg_path} with shape {self.bkg_3d.shape}")

        if os.path.isfile(sig_path):
            self.sig_3d = np.load(sig_path)
            print(f"Loaded sig_3d from {sig_path} with shape {self.sig_3d.shape}")

        if os.path.isfile(bkg_top_path):
            self.bkg_top = np.load(bkg_top_path)
            print(f"Loaded bkg_top from {bkg_top_path} with shape {self.bkg_top.shape}")

        if os.path.isfile(sig_top_path):
            self.sig_top = np.load(sig_top_path)
            print(f"Loaded sig_top from {sig_top_path} with shape {self.sig_top.shape}")

    # -------------------------------------------------------------
    # Step 6: Combine bkg & sig => X, y
    # -------------------------------------------------------------
    def _combine_sig_bkg(self) -> None:
        """
        Combine bkg & sig arrays => self.x, self.y.
        This method combines the 3D arrays of background and signal jets into a single array X,
        and creates corresponding labels y. It also combines the top-level 2D arrays if available.
        The y lanbels are one-hot encoded with two classes: background (0) and signal (1).
        """
        print("[_combine_sig_bkg] Combining bkg & sig => X, y... (and top-level data)")

        if (self.bkg_3d is None) or (self.sig_3d is None):
            print("  Missing bkg or sig 3D arrays. Skipping.")
            self.x = None
            self.y = None
        else:
            X = np.concatenate([self.bkg_3d, self.sig_3d], axis=0)
            labels = np.concatenate(
                [
                    np.zeros(len(self.bkg_3d), dtype=np.int32),
                    np.ones(len(self.sig_3d), dtype=np.int32),
                ]
            )
            y = labels.reshape(-1, 1).astype(np.float32)
            self.x = X
            self.y = y
            print("Combined X shape:", X.shape, " y shape:", y.shape)

        if (getattr(self, "bkg_top", None) is not None) and (
            getattr(self, "sig_top", None) is not None
        ):
            X_top = np.concatenate([self.bkg_top, self.sig_top], axis=0)
            if getattr(self, "y", None) is not None:
                y_top = self.y
            else:
                labels_top = np.concatenate(
                    [
                        np.zeros(len(self.bkg_top), dtype=np.int32),
                        np.ones(len(self.sig_top), dtype=np.int32),
                    ]
                )
                y_top = np.eye(2, dtype=np.float32)[labels_top]

            self.x_top = X_top
            self.y_top = y_top
            print(
                "Combined top-level X_top shape:",
                X_top.shape,
                " y_top shape:",
                y_top.shape,
            )
        else:
            print("  Missing bkg or sig 2D arrays. Skipping top-level combination.")

    # -------------------------------------------------------------
    # Step 7: Normalize (optional)
    # -------------------------------------------------------------
    def _normalize_data(self) -> None:
        """
        Normalize data and split into training and test sets.
        This method normalizes the 3D data (jets) and the top-level 2D data (jets) using a RobustScaler.
        It also splits the data into training and test sets, saving them in the processed folder.
        The normalization is done for both 3D and 2D data separately.
        Normalization is currently not done as one added a batch normalization layer in the model. 
        This is the method used in the SC4 algorithm.
        """
        print("[_normalize_data] Normalizing data...")

        # ---------- 3D normalization ----------
        if self.x is not None and self.y is not None:
            # Feature list for 3D data
            feature_list = [
                "pt",
                # "eta","phi",
                "vz",
                "pt_rel",
                "log_pt",
                "deta",
                "dphi",
                "isfilled",
                "mass",
                "puppiWeight",
            ] + [f"pdg_{p}" for p in self.pdg_order]

            # Perform normalization and splitting
            self.x_train, self.x_test, self.y_train, self.y_test, self.scaler = (
                self._norm_split(self.x, self.y, features=feature_list)
            )

            print("Setting self.x, self.y to training split (3D data)")
            self.x = self.x_train
            self.y = self.y_train

            # Save 3D data
            np.save(
                os.path.join(
                    self.processed_path, f"proc_train_{self.preproc_output_name}.npy"
                ),
                self.x,
            )
            np.save(
                os.path.join(
                    self.processed_path,
                    f"proc_labels_train_{self.preproc_output_name}.npy",
                ),
                self.y,
            )
            np.save(
                os.path.join(
                    self.processed_path, f"proc_test_{self.preproc_output_name}.npy"
                ),
                self.x_test,
            )
            np.save(
                os.path.join(
                    self.processed_path,
                    f"proc_labels_test_{self.preproc_output_name}.npy",
                ),
                self.y_test,
            )

            print("After normalization (3D), shapes:")
            print("  self.x:", self.x.shape, " self.y:", self.y.shape)
        else:
            print("  No 3D data found. Skipping 3D normalization.")

        # ---------- 2D (top-level) normalization ----------
        if (
            hasattr(self, "x_top")
            and hasattr(self, "y_top")
            and self.x_top is not None
            and self.y_top is not None
        ):
            top_level_list = ["pt", "eta", "phi", "mass"]

            X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
                self.x_top,
                self.y_top,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            # Use a separate scaler for top-level data if you like:
            self.scaler_top = RobustScaler()
            # X_train_top = self.scaler_top.fit_transform(X_train_top)
            # X_test_top = self.scaler_top.transform(X_test_top)

            # Save 2D data
            np.save(
                os.path.join(
                    self.processed_path,
                    f"proc_top_train_{self.preproc_output_name}.npy",
                ),
                X_train_top,
            )
            np.save(
                os.path.join(
                    self.processed_path,
                    f"proc_top_labels_train_{self.preproc_output_name}.npy",
                ),
                y_train_top,
            )
            np.save(
                os.path.join(
                    self.processed_path, f"proc_top_test_{self.preproc_output_name}.npy"
                ),
                X_test_top,
            )
            np.save(
                os.path.join(
                    self.processed_path,
                    f"proc_top_labels_test_{self.preproc_output_name}.npy",
                ),
                y_test_top,
            )

            print(
                f"After normalization (2D), train shape {X_train_top.shape}, test shape {X_test_top.shape}"
            )
            self.x_top = X_train_top
            self.y_top = y_train_top
        else:
            print("  No 2D top-level data found. Skipping 2D normalization.")

    def _plot_all_features(self, X_3d, features, stage="before") -> None:
        """
        Plots a histogram for each feature in features from the 3D array,
        using only the values for daughter 0 across all events.
        X_3d shape: (numEvents, numConstituents, numFeatures).

        Args:
            X_3d: 3D NumPy array of shape (numEvents, numConstituents, numFeatures).
            features: List of feature names to plot.
            stage: "before" or "after" normalization, used for labeling the plots.
        """
        for i, feat_name in enumerate(features):
            # Only take the 0th daughter across all events
            feat_values = X_3d[:, :, i].flatten()

            plt.figure()
            plt.hist(
                feat_values, bins=50, histtype="step", label=f"{feat_name} ({stage})"
            )
            plt.xlabel(feat_name)
            plt.yscale("log")
            plt.grid()
            plt.ylabel("Count")
            plt.title(
                f"Distribution of {feat_name} ({stage} normalization) for all constituents"
            )
            plt.legend()
            plt.show()
            if stage == "before":
                plt.savefig(
                    os.path.join(self.preprocessed_path, f"{feat_name}_{stage}.png")
                )
            else:
                plt.savefig(
                    os.path.join(self.processed_path, f"{feat_name}_{stage}.png")
                )
            plt.close()
        print(f"Plotted all features for {stage} normalization.")

    def load_normalized_data(self) -> None:
        """
        Loads the normalized data from the processed folder.
        It checks for the existence of the train and test files for both 3D and top-level data.
        If the files exist, it loads them into self.x and self.y for 3D data,
        and self.x_top and self.y_top for top-level data.
        If the files do not exist, it will not raise an error but will skip the loading.
        Note: This method assumes that the processed data has been saved in the expected format.
        """
        train_path = os.path.join(
            self.processed_path, f"train_{self.proc_output_name}.npy"
        )
        test_path = os.path.join(
            self.processed_path, f"test_{self.proc_output_name}.npy"
        )

        if os.path.isfile(train_path):
            self.x = np.load(train_path)
            print(f"Loaded x_train from {train_path} with shape {self.x.shape}")

        if os.path.isfile(test_path):
            self.x = np.load(test_path)
            print(f"Loaded x_test from {test_path} with shape {self.x.shape}")

        train_labels_path = os.path.join(
            self.processed_path, f"train_labels_{self.proc_output_name}.npy"
        )
        test_labels_path = os.path.join(
            self.processed_path, f"test_labels_{self.proc_output_name}.npy"
        )

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
        Normalize the 3D data X and split it into training and test sets. 
        This method flattens the 3D array, applies a mask to ignore missing constituents,
        and then uses a RobustScaler to normalize the specified features.
        Currently, it does not apply normalization as one added a batch normalization layer in the model.
        The data is plotted before and after normalization for visual inspection.

        Args:
            X: 3D NumPy array of shape (nEvents, nConstituents, nFeatures).
            y: Labels corresponding to the events in X.
            features: List of feature names to normalize.
        """

        # 1) Flatten and Mask the missing constituents with nan so that they are ignored in the scaling
        nEv, nC, nF = X.shape
        # We need to check if the pt value is -1 and if yes we set all features to np.nan
        mask = X[:, :, 0:1] == -1
        X = X.astype(np.float32)
        X = np.where(mask, np.nan, X)


        # mask = X == -1
        # np.where(np.sum(mask, axis=1) < mask.shape[-1], True, False)
        # X[mask] = np.nan
        X2d = X.reshape(nEv, nC * nF)

        # 2) Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X2d, y, test_size=self.test_size, random_state=self.random_state
        )

        # --- PLOT BEFORE SCALING ON TRAINING SUBSET ---
        print("\nPlotting all features before scaling (training set)...")
        X_train_3d_before = X_train.reshape(-1, nC, nF)
        self._plot_all_features(X_train_3d_before, features, stage="before")


        # 3) Figure out which features to normalize
        skip = (
            {"pt_rel"}
            | {"isfilled"}
            | {"puppiWeight"}
            | {f"pdg_{p}" for p in self.pdg_order}
        )
        feat_to_norm = [feat for feat in features if feat not in skip]
        feat_idx = [features.index(feat) for feat in feat_to_norm]

        # 4) Build column indices across all constituents
        col_idxs = sorted(c * nF + f for c in range(nC) for f in feat_idx)
        print(
            "\nColumn indices to normalize:",
            col_idxs[:8],
            "... (total",
            len(col_idxs),
            ")",
        )

        # 5) Scale only those columns
        scaler = RobustScaler()
        # X_train[:, col_idxs] = scaler.fit_transform(X_train[:, col_idxs])
        # X_test[:,  col_idxs] = scaler.transform(X_test[:,  col_idxs])

        # 6) reshape back to 3D
        # X_train.shape = (nEv, nC*nF) => (nEv, nC, nF)
        # X_test.shape  = (nEv, nC*nF) => (nEv, nC, nF)
        X_train_3d = X_train.reshape(-1, nC, nF)
        X_test_3d = X_test.reshape(-1, nC, nF)

        # 7) Check for NaN values and replace them with self.padding_value
        X_train_3d = np.nan_to_num(X_train_3d, nan=self.padding_value)
        X_test_3d = np.nan_to_num(X_test_3d, nan=self.padding_value)

        # Plot after scaling (on the training subset, for example)
        print("Plotting all features after scaling...")
        self._plot_all_features(X_train_3d, features, stage="after")

        return X_train_3d, X_test_3d, y_train, y_test, scaler

    # -------------------------------------------------------------
    # Step 7b: K-fold splitting
    # -------------------------------------------------------------
    def _split_kfold(self) -> None:
        """
        Split into k folds if requested.
        """
        if self.kfolds > 0 and self.train:
            print(f"[_split_kfold] Using {self.kfolds}-fold cross-validation...")
            skf = sklearn.model_selection.StratifiedKFold(
                n_splits=self.kfolds, shuffle=True, random_state=self.seed
            )
            # We have self.x (shape [n_samples, …]) and self.y (labels [n_samples])
            # Build the list of (train_idx, valid_idx) pairs:
            # self.kfolds = list(skf.split(self.x, self.y))

            # Convert back from one-hot to class targets since sklearn function does not
            # like one-hot targets.
            self.kfold_indices = list(skf.split(self.x, np.argmax(self.y, axis=-1)))


    # -------------------------------------------------------------
    # Step 8: Load test data
    # -------------------------------------------------------------
    def load_proc_test_data(self) -> None:
        """
        Load the test data (and top-level test data if analysis=True) from the processed folder.
        """
        test_path = os.path.join(
            self.processed_path, f"proc_test_{self.preproc_output_name}.npy"
        )
        test_labels_path = os.path.join(
            self.processed_path, f"proc_labels_test_{self.preproc_output_name}.npy"
        )

        if os.path.isfile(test_path):
            self.x = np.load(test_path)
            print(f"Loaded x_test from {test_path} with shape {self.x.shape}")

        if os.path.isfile(test_labels_path):
            self.y = np.load(test_labels_path)
            print(f"Loaded y_test from {test_labels_path} with shape {self.y.shape}")

        # If analysis is True, also load the top-level test data
        if self.analysis:
            top_test_path = os.path.join(
                self.processed_path, f"proc_top_test_{self.preproc_output_name}.npy"
            )
            top_test_labels_path = os.path.join(
                self.processed_path,
                f"proc_top_labels_test_{self.preproc_output_name}.npy",
            )

            if os.path.isfile(top_test_path):
                self.x_top = np.load(top_test_path)
                print(
                    f"Loaded x_top_test from {top_test_path} with shape {self.x_top.shape}"
                )

            if os.path.isfile(top_test_labels_path):
                self.y_top = np.load(top_test_labels_path)
                print(
                    f"Loaded y_top_test from {top_test_labels_path} with shape {self.y_top.shape}"
                )

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------
    def show_details(self) -> None:
        """
        Print some info about current data shapes.
        """
        print("Data shape X:", None if self.x is None else self.x.shape)
        print("Data shape y:", None if self.y is None else self.y.shape)
        print("Number of folds:", self.kfolds)


