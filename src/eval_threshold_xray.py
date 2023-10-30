import torch
import pandas as pd
import cxr_dataset as CXR
from torchvision import transforms, models, utils
from torch.utils.data import Dataset, DataLoader
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np

import torch.optim as optim

# calc preds in batches of 16, can reduce if your GPU has less RAM
def make_pred_multi_label_threshold(PATH_TO_MODEL, WEIGHT_DECAY, BATCH_SIZE, dataloaders, PRED_LABEL, RESULTS_DIRECTORY, DATASET_NAME, mode=None):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    device = torch.device("cuda")

    model = models.densenet121()
    checkpoint = torch.load(PATH_TO_MODEL)
    model = checkpoint['model']
    epoch = checkpoint['epoch']
    loss = checkpoint['best_loss']
    LR = checkpoint['LR']
    optimizer = optim.SGD( filter(lambda p: p.requires_grad,model.parameters()),lr=LR,momentum=0.9,weight_decay=WEIGHT_DECAY)
    model.eval()
    model.to(device)

    # set model to eval mode; required for proper predictions given use of batchnorm
    for MODE in ["Threshold", "test"]:
        # create empty dfs
        pred_df = pd.DataFrame(columns=["path"])
        bi_pred_df = pd.DataFrame(columns=["path"])
        true_df = pd.DataFrame(columns=["path"])

        if MODE == "Threshold":
            loader = dataloaders["val"]
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if MODE == "test":
            loader = dataloaders["test"]
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])

            Eval = pd.read_csv(RESULTS_DIRECTORY + DATASET_NAME + "_Threshold_pretrained.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],                    
                    Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pleural_Thickening"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]]]

        for i, data in enumerate(loader):
            inputs, labels, item = data

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()).float()

            true_labels = labels.cpu().data.numpy()
            batch_size = true_labels.shape

            outputs = model(inputs)
            probs = outputs.cpu().data.numpy()

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                truerow["path"] = item[j]
                thisrow["path"] = item[j]
                if MODE == "test":
                    bi_thisrow["path"] = item[j]

                    # iterate over each entry in prediction vector; each corresponds to
                    # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    if MODE == "test":
                       bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                pred_df = pred_df.append(thisrow, ignore_index=True)
                true_df = true_df.append(truerow, ignore_index=True)
                if MODE == "test":
                    bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)

            if (i % 200 == 0):
                print(str(i * BATCH_SIZE))


        for column in true_df:
            if column not in PRED_LABEL:
                continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            
            thisrow = {}
            thisrow['label'] = column
            
            if MODE == "test":
                bi_pred = bi_pred_df["bi_" + column]            
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            try:

                if MODE == "test":
                    thisrow['auc'] = sklm.roc_auc_score(
                        actual.to_numpy().astype(int), pred.to_numpy())

                    thisrow['auprc'] = sklm.average_precision_score(
                        actual.to_numpy().astype(int), pred.to_numpy())
                else:

                    p, r, t = sklm.precision_recall_curve(actual.to_numpy().astype(int), pred.to_numpy())
                    # Choose the best threshold based on the highest F1 measure
                    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                    bestthr = t[np.where(f1 == max(f1))]

                    thrs.append(bestthr)
                    thisrow['bestthr'] = bestthr[0]


            except BaseException:
                print("can't calculate auc for " + str(column))

            if MODE == "Threshold":
                Eval_df = Eval_df.append(thisrow, ignore_index=True)

            if MODE == "test":
                TestEval_df = TestEval_df.append(thisrow, ignore_index=True)

        pred_df.to_csv(RESULTS_DIRECTORY + DATASET_NAME + "_preds_pretrained.csv", index=False)
        true_df.to_csv(RESULTS_DIRECTORY + DATASET_NAME + "_True_pretrained.csv", index=False)


        if MODE == "Threshold":
            Eval_df.to_csv(RESULTS_DIRECTORY + DATASET_NAME + "_Threshold_pretrained.csv", index=False)

        if MODE == "test":
            TestEval_df.to_csv(RESULTS_DIRECTORY + DATASET_NAME + "_TestEval_pretrained.csv", index=False)
            bi_pred_df.to_csv(RESULTS_DIRECTORY + DATASET_NAME + "_bipred_pretrained.csv", index=False)

    
    print("AUC ave:", TestEval_df['auc'].sum() / 14.0)

    print("done")
