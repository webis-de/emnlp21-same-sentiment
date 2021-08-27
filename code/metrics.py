# https://github.com/huggingface/transformers/blob/9e9a1fb8c75e2ef00fea9c4c0dc511fc0178081c/src/transformers/data/metrics/__init__.py
from itertools import product

import numpy as np
import sklearn.utils

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from scipy.special import expit
from scipy.stats import pearsonr, spearmanr


def is_sklearn_available():
    return True


# ---------------------------------------------------------------------------


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


# ---------------------------------------------------------------------------


def sameness_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    # return {"mcc": matthews_corrcoef(labels, preds)}
    # return acc_and_f1(preds, labels)

    if task_name == "sent-5":
        return {
            "acc": simple_accuracy(preds, labels),
            "f1-micro": f1_score(y_true=labels, y_pred=preds, average="micro"),
            "f1-macro": f1_score(y_true=labels, y_pred=preds, average="macro"),
            "f1-weighted": f1_score(y_true=labels, y_pred=preds, average="weighted"),
            **pearson_and_spearman(preds, labels),
        }
    elif task_name in ("sent-b", "same-b"):
        return {
            **acc_and_f1(preds, labels),
            **pearson_and_spearman(preds, labels),
            "class_report": classification_report(
                y_true=labels,
                y_pred=preds,
                output_dict=True,
                labels=[0, 1],
                target_names=["not same", "same"],
            ),
        }
    elif task_name in ("sent-r", "same-r"):
        # TODO: how to better do this ...
        # preds2 = expit(preds).round().astype("int32")
        preds2 = preds.round().astype("int32")
        # labels = labels.astype("float")
        # preds = preds.astype("float")
        # float can not use average="binary" in f1_score
        return {
            **acc_and_f1(preds2, labels),
            **pearson_and_spearman(preds, labels),
        }
    else:
        raise KeyError(task_name)


# ---------------------------------------------------------------------------
# old: baseline


def compute_metrics_old(conf_mat, precision=3, dump=True):
    conf_mat = np.array(conf_mat)
    tn, fp, fn, tp = conf_mat.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)

    if dump:
        print("{:>10}: {:.{prec}f}".format("accuracy", acc, prec=precision))
        print("{:>10}: {:.{prec}f}".format("precision", prec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("recall", rec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("f1-score", f1, prec=precision))

    return prec, rec, f1, acc


def compute_metrics(labels, preds, precision=3, averaging="macro", dump=True):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, labels=[0, 1], average=averaging)
    rec  = recall_score(labels, preds, labels=[0, 1], average=averaging)
    f1 = f1_score(labels, preds, labels=[0, 1], average=averaging)
    cm = confusion_matrix(labels, preds)

    if dump:
        print("CM:", cm.ravel(), "\n[tn, fp, fn, tp]")
        print("{:>10}: {:.{prec}f}".format("accuracy", acc, prec=precision))
        print("{:>10}: {:.{prec}f}".format("precision", prec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("recall", rec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("f1-score", f1, prec=precision))

    return prec, rec, f1, acc, cm


def heatconmat(y_test, y_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context('talk')
    plt.figure(figsize=(9, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='gist_earth_r',
                yticklabels=sorted(np.unique(y_test)))
    plt.show()


# see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix
class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels, title=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.title = title

    def plot(self, include_values=True, cmap='viridis', show_colorbar=True,
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='vertical'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2f' for a normalized matrix, and
            'd' for a unnormalized matrix.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        sklearn.utils.check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if self.title:
            fig.suptitle(self.title)

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() - cm.min()) / 2.
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        if show_colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(y_true, y_pred, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None, title=None,
                          include_colorbar=True,
                          cmap='viridis', ax=None):
    """Plot Confusion Matrix.
    Read more in the :ref:`User Guide <confusion_matrix>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target values.
    y_pred : array-like of shape (n_samples,)
        Prediction values.
    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : array-like of shape (n_classes,), default=None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred`
        will be used.
    include_values : bool, default=True
        Includes values in confusion matrix.
    xticks_rotation : {'vertical', 'horizontal'} or float, \
                        default='vertical'
        Rotation of xtick labels.
    values_format : str, default=None
        Format specification for values in confusion matrix. If `None`,
        the format specification is '.2f' for a normalized matrix, and
        'd' for a unnormalized matrix.
    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.
    ax : matplotlib Axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created.
    Returns
    -------
    display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    """
    sklearn.utils.check_matplotlib_support("plot_confusion_matrix")

    if normalize not in {'true', 'pred', 'all', None}:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels)

    if display_labels is None:
        if labels is None:
            raise ValueError("Missing labels!")
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels,
                                  title=title)
    return disp.plot(include_values=include_values,
                     values_format=values_format, show_colorbar=include_colorbar,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)


def report_training_results(y_test, y_pred, name=None, heatmap=True, metrics=True):
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()
    if metrics:
        # compute_metrics(confusion_matrix(y_test, y_pred))
        compute_metrics(y_test, y_pred)
    if heatmap:
        heatconmat(y_test, y_pred)
    print()
    print('Accuracy: ', round(accuracy_score(y_test, y_pred), 3), '\n')  #

    print('Report{}:'.format("" if not name else " for [{}]".format(name)))
    print(classification_report(y_test, y_pred))

    f1_dic = {}
    f1_dic['macro'] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average='macro'), 3)
    f1_dic['micro'] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average='micro'), 3)
    return f1_dic



# ---------------------------------------------------------------------------
