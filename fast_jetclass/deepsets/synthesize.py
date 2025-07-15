# Script that synthesizes a given deepsets model and then returns the performance
# performance metrics for the synthesis.
import sys
import io
import os
import collections.abc
import json
import contextlib
import matplotlib.pyplot as plt
from sklearn import metrics

import numpy as np
import tensorflow as tf
import hls4ml
from hls4ml.model.profiling import get_ymodel_keras, numerical
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import qkeras
from qkeras.quantizers import quantized_bits

# np.random.seed(12)
# tf.random.set_seed(12)

from tensorflow import keras
import tensorflow.keras.layers as KL

from fast_jetclass.util import util
from fast_jetclass.util import plots
from fast_jetclass.util.terminal_colors import tcols
from fast_jetclass.data.hls4ml150 import HLS4MLData150
from fast_jetclass.data.FullJetData import FullJetData
from fast_jetclass.data.SC8Data import SC8Data


def main(args, synth_config: dict):
    util.device_info()
    synthesis_dir = util.make_output_directories(args.model_dir, "synthesis")

    print(tcols.OKGREEN + "\nIMPORTING DATA AND MODEL\n" + tcols.ENDC)
    root_dir = os.path.dirname(os.path.abspath(args.model_dir))
    hyperparams = util.load_hyperparameter_file(root_dir)
    valid_data = util.import_data(hyperparams["data_hyperparams"], train=False)

    # Take the first 6000 events to do the diagnosis of the synthesis.
    # More are not really needed and it increases the runtime of this script by a lot.
    valid_data.x = valid_data.x
    valid_data.y = valid_data.y

    #input_quantizer = quantized_bits(bits=12, integer=8, symmetric=0, alpha=1)
    #valid_data.x = input_quantizer(valid_data.x.astype(np.float32)).numpy()

    #valid_data.shuffle_constituents(args.seed)
    model = import_model(args.model_dir, hyperparams)

    print(tcols.OKGREEN + "\nCONFIGURING SYNTHESIS\n" + tcols.ENDC)
    hls4ml_config = hls4ml.utils.config_from_keras_model(model, granularity="name")
 
    deep_dict_update(hls4ml_config, synth_config)


    model_activations = get_model_activations(model)
    print(tcols.HEADER + "Model activations: " + tcols.ENDC)
    print(model_activations)
    # Set the model activation function rounding and saturation modes.
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        layers=model_activations,
        rounding_mode="AP_RND",
        saturation_mode="AP_SAT",
    )

    if args.diagnose:
        print("\nSetting Trace=True for all layers...")
        for layer in hls4ml_config["LayerName"].keys():
            hls4ml_config["LayerName"][layer]["Trace"] = True

        # 🔍 Confirm it's applied
        for layer, conf in hls4ml_config["LayerName"].items():
            print(f"Layer: {layer} -> Trace: {conf.get('Trace', False)}")

    print(tcols.HEADER + "Configuration for hls4ml: " + tcols.ENDC)
    print(json.dumps(hls4ml_config, indent=4, sort_keys=True))


    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        project_name="L1TSC8NGJetModel",
        hls_config=hls4ml_config,
        clock_period=2.8, #1/360MHz = 2.8ns
        part='xcvu9p-flga2104-2L-e',
        output_dir=synthesis_dir,
        io_type="io_parallel",
        backend="Vitis",
    )
         
    hls_model.compile()
    if args.diagnose:
        print(tcols.OKGREEN + "\nRUNNING MODEL DIAGNOSTICS" + tcols.ENDC)
        run_trace(model, hls_model, valid_data.x, synthesis_dir, sample_idx=args.sample)
        profile_model(model, hls_model, valid_data.x, synthesis_dir)
    hls_model.write()

    print(tcols.OKGREEN + "\nTESTING MODEL PERFORMANCE\n" + tcols.ENDC)
    print(tcols.HEADER + f"\nRunning inference for {args.model_dir}" + tcols.ENDC)
    y_pred = run_inference(model, valid_data)
    acc = calculate_accuracy(y_pred, valid_data.y)
    y_pred = hls_model.predict(valid_data.x)
    acc_synth = calculate_accuracy(y_pred, valid_data.y)

    plots_dir = os.path.join(synthesis_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _= plots.roc_curves(plots_dir, y_pred, valid_data.y)
    _= roc_curves_comparison(
        plots_dir,
        y_pred_qkeras=run_inference(model, valid_data),
        y_pred_hls4ml=hls_model.predict(valid_data.x),
        y_test=valid_data.y
    )
    print()
    print(f"Accuracy model: {acc:.3f}")
    print(f"Accuracy synthed model: {acc_synth:.3f}")
    print(f"Accuracy ratio: {acc_synth/acc:.3f}")



    #I want to load data and just save the output of the hls4ml model to a file.
    henry_path = "/work/aserinke/SC8_Data/Henry_Data/inputs_physical.npy"
    henry_data = np.load(henry_path)
    henry_data = henry_data.astype(np.float32)
    print("Loaded Henry data keys:", henry_data.shape)
    henry_scores = hls_model.predict(henry_data)
    henry_scores = henry_scores.astype(np.float32)
    henry_scores_path = os.path.join(synthesis_dir, f"hls_scores_n8.npy")
    np.save(henry_scores_path, henry_scores)
    print("Saved Henry scores to:", henry_scores_path)
    print("Standalone scores shape:", henry_scores.shape)

#Function which makes the ROC curves for the model and the synthesized model.
def roc_curves_comparison(outdir: str,
                          y_pred_qkeras: np.ndarray,
                          y_pred_hls4ml: np.ndarray,
                          y_test: np.ndarray):
    """
    Plot ROC curves for the 2-prong class, comparing:
      • raw scores (softmax output for class 1)
      • “ratio” scores (p1 / (p0 + p1))
    for both QKeras and HLS4ML versions.
    """
    # colours & linestyles
    framework_colors = {
        "QKeras": "#1f77b4",   # blue
        "HLS4ML": "#ff7f0e",   # orange
    }
    curve_styles = {
        "raw":   {"linestyle": "-",  "label_fmt": "{} 2prong"},
    }

    # baseline TPR axis (for interpolation)
    tpr_baseline = np.linspace(0.025, 0.99, 200)

    os.makedirs(outdir, exist_ok=True)

    for framework, y_pred in (("QKeras", y_pred_qkeras),
                              ("HLS4ML", y_pred_hls4ml)):

        color = framework_colors[framework]
        # make QKeras curves semi‐transparent
        alpha = 0.5 

        for kind, cfg in curve_styles.items():
            # pick scores
            if kind == "raw":
                # For a binary classifier, use the predicted probability directly.
                scores = y_pred  

            # compute ROC
            fpr, tpr, _ = metrics.roc_curve(y_test, scores)
            auc = metrics.auc(fpr, tpr)

            # interpolate for downstream tools
            fpr_smooth = np.interp(tpr_baseline, tpr, fpr)

            # save arrays
            fpr_smooth.astype("float32").tofile(
                os.path.join(outdir, f"fpr_{framework}_{kind}.dat")
            )
            tpr_baseline.astype("float32").tofile(
                os.path.join(outdir, f"tpr_{framework}_{kind}.dat")
            )

            # find FPR @ 80% TPR
            idx_80 = plots.find_nearest(tpr, 0.8)
            fpr80 = fpr[idx_80]

            # plot
            plt.plot(
                tpr,
                fpr,
                color=color,
                linestyle=cfg["linestyle"],
                alpha=alpha,
                label=(cfg["label_fmt"].format(framework)
                       + f": AUC={auc*100:.1f}%, FPR@80%TPR={fpr80:.3f}")
            )

    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.ylim(1e-3, 1)
    plt.xlim(0.8, 1)
    plt.grid()
    plt.semilogy()
    plt.legend(prop={"size": 11})
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves_comparison.pdf"))
    plt.close()
    print(f"ROC comparison plot saved to {outdir}.")


def import_model(model_dir: str, hyperparams: dict):
    """Imports the model from a specified path. Model is saved in tf format."""
    print(tcols.HEADER + "\n\nDeepSets hyperparams and architecture: " + tcols.ENDC)
    print(json.dumps(hyperparams["model_hyperparams"], indent=4, sort_keys=True))
    model = keras.models.load_model(
        model_dir,
        compile=False,
        custom_objects={
            "QDense": qkeras.QDense,
            "QActivation": qkeras.QActivation,
            "quantized_bits": qkeras.quantized_bits,
            "PruneLowMagnitude": pruning_wrapper.PruneLowMagnitude,
        },
    )
    if "pruning_rate" in hyperparams["training_hyperparams"].keys():
        if hyperparams["training_hyperparams"]["pruning_rate"] > 0:
            model = strip_pruning(model)

    model.summary(expand_nested=True)

    return model


def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray):
    """Computes accuracy for a model's predictions, given the true labels y_true."""
    acc = keras.metrics.CategoricalAccuracy()
    acc.update_state(y_true, y_pred)

    return acc.result().numpy()


def run_inference(model: keras.Model, data: FullJetData | SC8Data):
    """Computes predictions of a binary model and saves them to numpy files."""
    # For a binary model with one output, no softmax is needed.
    y_pred = model.predict(data.x).flatten()
    return y_pred


def profile_model(
    model: keras.Model, hls_model: hls4ml.model, data: np.ndarray, outdir
):
    """Profile the hls4ml model to see the bit width of every layer.

    The plots in this function show the distribution of weights in the network.
    """
    fig1, fig2, fig3, fig4 = numerical(
        model=model, hls_model=hls_model, X=data[:5000]
        )

    fig1.savefig(os.path.join(outdir, "fig1"))
    fig2.savefig(os.path.join(outdir, "fig2"))
    fig3.savefig(os.path.join(outdir, "fig3"))
    fig4.savefig(os.path.join(outdir, "fig4"))
    hls4ml.utils.plot_model(
        hls_model,
        show_shapes=True,
        show_precision=True,
        to_file=os.path.join(outdir, "model_plot.png"),
    )


def run_trace(model: keras.Model, hls_model: hls4ml.model, data: np.ndarray, outdir, sample_idx: int = 0):
    """Shows output of every layer given a certain sample.

    This is used to compute the outputs in every layer for the hls4ml firmware model
    against the qkeras model. The outputs of the quantized layers is not quantized
    itself in the QKERAS model but it is in hls4ml. A big difference in outputs is
    indicative that the precision of these outputs should be set higher manually
    in hls4ml.
    """
    # ---- run trace for one event ------------------------------------
    x_one   = data[sample_idx : sample_idx + 1]          # keep batch dim
    _, hls_t = hls_model.trace(x_one)
    keras_t  = get_ymodel_keras(model, x_one)

    # ---- console diff summary ---------------------------------------
    rows = []
    for layer in model.layers:
        if layer.name == "input_layer":
            continue
        ref = keras_t[layer.name][0]
        dut = hls_t[layer.name][0]
        err = np.max(np.abs(ref - dut))
        rows.append((layer.name, ref.shape, err))

    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\n🔍  layer-wise max|Δ| for sample {sample_idx}")
    print("{:<24s} {:<14s} {:>10s}".format("layer", "shape", "max|Δ|"))
    print("-"*50)
    for name, shape, err in rows:
        flag = " !!" if err > 1e-3 else ""
        print(f"{name:<24s} {str(shape):<14s} {err:10.4e}{flag}")

    # ---- CSV for later -------------------------------------
    import csv, os
    csv_path = os.path.join(outdir, f"trace_diff_sample{sample_idx}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "shape", "max_abs_diff"]); w.writerows(rows)
    print(f"📄  full diff table written to {csv_path}")


    # Show the weights of the network only for 3 of the samples, as defined below.
    sample_numbers = [sample_idx]

    # Take just the first 100 events of the data set.
    hls4ml_pred, hls4ml_trace = hls_model.trace(data[:100])
    keras_trace = get_ymodel_keras(model, data[:100])

    # Write the weights of the hls4ml and qkeras networks for the 3 specified samples.
    trace_file_path = os.path.join(outdir, "trace_output.log")
    with open(trace_file_path, "w") as trace_file:
        for sample_number in sample_numbers:
            for layer in model.layers:
                if layer.name == "input_layer":
                    continue
                trace_file.write(f"Layer output HLS4ML for {layer.name}")
                trace_file.write(str(hls4ml_trace[layer.name][sample_number]))
                trace_file.write(f"Layer output KERAS for {layer.name}")
                trace_file.write(str(keras_trace[layer.name][sample_number]))
                trace_file.write("\n")

    print(tcols.OKGREEN)
    print(f"Wrote trace for samples {sample_numbers} to file {trace_file_path}.")
    print(tcols.ENDC)


def get_model_activations(model: keras.Model):
    """Looks at the layers in a model and returns a list with all the activations.

    This is done such that the precision of the activation functions is set separately
    for the synthesis on the FPGA.
    """
    model_activations = []
    for layer in model.layers:
        if "activation" in layer.name:
            model_activations.append(layer.name)


    return model_activations


def deep_dict_update(d, u):
    """Updates a deep dictionary on its deepsets entries."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


@contextlib.contextmanager
def nostdout():
    """Suppresses the terminal output of a function."""
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout
