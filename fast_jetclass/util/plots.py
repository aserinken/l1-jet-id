# Methods to plot the results of a model tested on some data.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def loss_vs_epochs(
    outdir: str,
    train_loss: np.ndarray,
    valid_loss: np.ndarray,
    plot_name: str = "loss_epochs",
):
    """Plots the loss for each epoch for the training and validation data
    and saves it to the same directory the model is saved in.
    """
    plt.plot(train_loss, color="gray", label="Training Loss")
    plt.plot(valid_loss, color="navy", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.text(
        0,
        np.max(train_loss),
        f"Min: {np.min(valid_loss):.2e}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, plot_name + ".pdf"))
    plt.close()
    print(f"Loss vs epochs plot saved to {outdir}.")


def accuracy_vs_epochs(outdir: str, train_acc: np.ndarray, valid_acc: np.ndarray):
    """Plots the accuracy for each epoch for the training and validation data
    and saves it to the same directory the model is saved in.
    """
    plt.plot(train_acc, color="gray", label="Training Accuracy")
    plt.plot(valid_acc, color="navy", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.text(
        0,
        np.max(train_acc),
        f"Max: {np.max(valid_acc):.2e}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, "accuracy_epochs.pdf"))
    plt.close()
    print(f"Accuracy vs epochs plot saved to {outdir}.")


def find_nearest(array: np.ndarray, value: float):
    """Finds the index of the nearest value in an array to a given value."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def roc_curves(outdir: str, y_pred: np.ndarray, y_test: np.ndarray):
    """Plot the ROC curve for a binary classifier."""
    # Ensure y_pred is 2D: if it's 1D, expand it for consistency.
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] == 1:
        y_pred_bin = y_pred.flatten()
        y_test_bin = y_test.flatten()
        fpr, tpr, thr = metrics.roc_curve(y_test_bin, y_pred_bin)
        auc = metrics.auc(fpr, tpr)
        np.array(fpr, dtype="float32").tofile(os.path.join(outdir, "fpr_binary.dat"))
        np.array(tpr, dtype="float32").tofile(os.path.join(outdir, "tpr_binary.dat"))
        # Find the FPR at TPR = 0.8 using interpolation.
        fpr_at_80 = np.interp(0.8, tpr, fpr)
        plt.plot(
            tpr,
            fpr,
            color="#785EF0",
            label=f"AUC = {auc*100:.1f}%; FPR @ 80% TPR: {fpr_at_80:.3f}",
        )
        # Interpolate to a common baseline tpr
        tpr_baseline = np.linspace(0.025, 0.99, 100)
        fpr_baseline = np.interp(tpr_baseline, tpr, fpr)
        fprs = [fpr_baseline]
        aucs = [auc]
        fprs_at_tprs = [np.interp(0.8, tpr, fpr)]
    else:
        # Use the original two-class loop if needed
        labels = ["background", "2prong"]
        cols = ["#648FFF", "#785EF0"]
        tpr_baseline = np.linspace(0.025, 0.99, 100)
        fprs = []
        aucs = []
        fprs_at_tprs = []
        for idx, label in enumerate(labels):
            fpr, tpr, thr = metrics.roc_curve(y_test[:, idx], y_pred[:, idx])
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            fpr_baseline = np.interp(tpr_baseline, tpr, fpr)
            fprs.append(fpr_baseline)
            fpr_baseline.astype("float32").tofile(os.path.join(outdir, f"fpr_{label}.dat"))
            tpr_baseline.astype("float32").tofile(os.path.join(outdir, f"tpr_{label}.dat"))
            fpr_at_80 = np.interp(0.8, tpr, fpr)
            fprs_at_tprs.append(fpr_at_80)
            plt.plot(
                tpr,
                fpr,
                color=cols[idx],
                label=f"{label}: AUC = {auc*100:.1f}%; FPR @ 80% TPR: {fpr_at_80:.3f}",
            )
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.ylim(0.001, 1)
    plt.semilogy()
    plt.legend(prop={"size": 11})
    plt.savefig(os.path.join(outdir, "roc_curves.pdf"))
    plt.close()
    print(f"ROC curves plot saved to {outdir}.")
    
    return fprs, tpr_baseline, aucs, fprs_at_tprs


def roc_curves_uncert(
    tpr: np.ndarray,
    fprs: np.ndarray,
    fprs_errs: np.ndarray,
    aucs: np.ndarray,
    aucs_errs: np.ndarray,
    fats: np.ndarray,
    fats_errs: np.ndarray,
    outdir: str,
):
    """Plots ROC curves given fprs and tprs for each class."""
    # If we have only 1 set of fprs, then we're in binary mode:
    if len(fprs) == 1:
        labels = ["signal"]
        cols = ["#785EF0"]
    else:
        labels = ["background", "2prong"]
        cols = ["#648FFF", "#785EF0"]

    tpr_baseline = np.linspace(0.025, 0.99, 100)
    for idx, label in enumerate(labels):
        plt.plot(
            tpr,
            fprs[idx],
            color=cols[idx],
            label=f"{label}: AUC = {aucs[idx]*100:.1f}%; FAT: {fats[idx]:.4f} $\\pm$ {fats_errs[idx]:.4f}",
        )
        plt.fill_between(
            tpr,
            fprs[idx] - fprs_errs[idx],
            fprs[idx] + fprs_errs[idx],
            color=cols[idx],
            alpha=0.5,
        )

    plt.xlabel("True Positive Rate", fontsize=11)
    plt.ylabel("False Positive Rate", fontsize=11)
    plt.ylim(0.001, 1)
    plt.semilogy()
    plt.legend(prop={"size": 11})
    plt.savefig(os.path.join(outdir, "roc_curves.pdf"))
    plt.close()
    print(f"ROC curves plot saved to {outdir}.")


def dnn_output(outdir: str, y_pred: np.ndarray):
    """Plots the output of the last part (fc) of the interaction network."""
    bins = np.linspace(0.0, 1.0, 20)
    # Ensure y_pred is 2D - if it's 1D, reshape it.
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] == 1:
        plt.hist(y_pred.flatten(), bins, label="Signal probability", histtype="step", color="#785EF0")
    else:
        labels = ["background", "2prong"]
        cols = ["#648FFF", "#785EF0"]
        for idx, label in enumerate(labels):
            plt.hist(y_pred[:, idx], bins, label=label, histtype="step", color=cols[idx])
    plt.semilogy()
    plt.xlabel("$f_c(x)$ DNN Output")
    plt.legend()
    plt.savefig(os.path.join(outdir, "fc_output_histos.pdf"))
    plt.close()
    print(f"DNN output histograms plot saved to {outdir}.")