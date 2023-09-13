# yhc

# plot and display
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

color_list = ['aqua', 'darkorange', 'cornflowerblue',  'm', 'Orange', 'purple', 'Blue',  'Red', "y",  ]

# binary roc curve
def plot_roc_curve(checkpoint_folder, classes_name, preds, targets):
    o = softmax(preds, axis=1)  # output

    fpr, tpr, thresholds = roc_curve(targets, o[:, 1])
    auc_ = roc_auc_score(targets, o[:, 1])
    # auc__ = auc(fpr,tpr)  # only compute positive sample's auc
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % auc_, lw=1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(str(checkpoint_folder)+r'/roc_curve.png', dpi=800)
    # plt.show()
    plt.close()


# multiclass roc curve
def plot_roc_curve_multiclass(checkpoint_folder, classes_name, preds, targets):
    encoder = OneHotEncoder(sparse=False)
    o = softmax(preds, axis=1)  # output
    # o_ = np.eye(o.shape[1])[np.argmax(o, axis=1)]  # one hot
    # o_ = np.array(o_).reshape(-1, 1)
    # o_onehot = encoder.fit_transform(o_)
    targets_ = np.array(targets).reshape(-1, 1)
    targets_onehot = encoder.fit_transform(targets_)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, c in enumerate(classes_name):
        fpr[c], tpr[c], _ = roc_curve(targets_onehot[:, i], o[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])
    fpr["micro"], tpr["micro"], _ = roc_curve(targets_onehot.ravel(), o.ravel(), )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    colors = color_list[:len(classes_name)]
    # colors = cycle(color_list[:len(classes_name)])
    for (i, c), color in zip(enumerate(classes_name), colors):
        plt.plot(fpr[c], tpr[c], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                       ''.format(c, roc_auc[c]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(str(checkpoint_folder)+r'/roc_curve1.png', dpi=800)
    # plt.show()
    plt.close()


# plot Confusion matrix
def plot_confusion_matrix(checkpoint_folder, classes, cm,):
    cm = np.array(cm)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=300)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=14, va='center', ha='center')


    plt.imshow(cm_normalized, interpolation='nearest', cmap='viridis')
    plt.title('Confusion Matrix',)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=45)
    plt.yticks(xlocations, classes, )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(checkpoint_folder)+r'/Confusion_matrix.png', dpi=800)
    plt.close()


if __name__ == "__main__":
    checkpoint_folder = r"D:\yhc\MoCo-CXR-main\results\paper_3_moco_LoRA_4_multimodel\moco_model_and_datasets_20230623-131937"
    # classes = ["COVID", "NORMAL"]  # 11G 08G
    classes = ["COVID", "Lung_Opacity", "NORMAL", "Viral Pneumonia"]  # 08G
    # # classes = ["disease", "normal"]  # shenzhen
    # # classes = ["Lung_Opacity", "Normal"]  # RSNA
    # cm = []
    # plot_confusion_matrix(checkpoint_folder, classes, cm)

    preds = []
    targets = []

    plot_roc_curve_multiclass(checkpoint_folder, classes, preds, targets)
    print()