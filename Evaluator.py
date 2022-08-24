from sklearn.metrics import confusion_matrix,precision_score
from sklearn.metrics import recall_score,f1_score,accuracy_score
import pandas as pd
from utils import *
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from Models import AlephBERTModel
class Evaluator:
    def __init__(self, X_test, Y_test):
        self.X_test = pd.DataFrame(X_test)
        self.Y_test= pd.DataFrame(Y_test)
        print(f"test_df has {len(self.Y_test)} rows")

    def get_true_labels(self):
        return self.Y_test[["index", "label"]]

    def show_errors(self, models, n, correct_label_name):
        correct_label = classes[correct_label_name]
        for algo in models:
            print(f"showing errors of {algo} with correct label {correct_label_name}: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            predicted_labels = models[algo].predict(self.X_test.ReviewMainTxt)
            pred_df = self.Y_test.copy()
            pred_df.loc[:,"pred_label"]=predicted_labels
            pred_df.loc[:,"ReviewMainTxt"]=self.X_test.ReviewMainTxt
            pred_df.loc[:,"url"]=self.X_test.url
            wrong_df=pred_df[pred_df.pred_label!=pred_df.label]
            wrong_df=wrong_df[wrong_df.label==correct_label]
            print(f"overall, there are {len(wrong_df)} instances with wrong sentiment")
            num_shown=min(n, len(wrong_df))
            for i in range(0, num_shown):
                cur_row=wrong_df.iloc[i]
                print(f"true:{inv_classes[cur_row.label]}, predicted:{inv_classes[cur_row.pred_label]} url: {cur_row.url} ------------------------------- \n{cur_row.ReviewMainTxt}")

    def show_correct(self, models, n, correct_label_name):
        correct_label = classes[correct_label_name]
        for algo in models:
            print(f"showing errors of {algo} with correct label {correct_label_name}: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            predicted_labels = models[algo].predict(self.X_test.ReviewMainTxt)
            pred_df = self.Y_test.copy()
            pred_df.loc[:,"pred_label"]=predicted_labels
            pred_df.loc[:,"ReviewMainTxt"]=self.X_test.ReviewMainTxt
            pred_df.loc[:,"url"]=self.X_test.url
            correct_df=pred_df[pred_df.pred_label == pred_df.label]
            correct_df= correct_df[correct_df.label == correct_label]
            print(f"overall, there are {len(correct_df)} instances with wrong sentiment")
            num_shown=min(n, len(correct_df))
            for i in range(0, num_shown):
                cur_row=correct_df.iloc[i]
                print(f"true:{inv_classes[cur_row.label]}, predicted:{inv_classes[cur_row.pred_label]} url: {cur_row.url} ------------------------------- \n{cur_row.ReviewMainTxt}")

    def plot_AlephBERT_matrix(self,y_true, y_pred, labels=5):
        cm = confusion_matrix(y_true, y_pred)
        display_labels = []
        for i in range(labels):
            display_labels.append(i)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        return disp.plot(
            include_values=True,
            cmap="viridis",
            ax=None,
            xticks_rotation="horizontal",
            values_format=None,
            colorbar=True,
        )

    def evaluate(self, models):
        for algo in models:
            print(f"Evaluating {algo}:")
            predicted_labels = models[algo].predict(self.X_test.ReviewMainTxt)
            prec_mic = precision_score(self.Y_test, predicted_labels, average="micro")

            rec_mic = recall_score(self.Y_test, predicted_labels, average="micro")

            f1_mic = f1_score(self.Y_test, predicted_labels, average="micro")
            print(f"Micro precision:{prec_mic}, recall:{rec_mic}, f1:{f1_mic}")
            prec_mac = precision_score(self.Y_test, predicted_labels, average="macro")

            rec_mac = recall_score(self.Y_test, predicted_labels, average="macro")

            f1_mac = f1_score(self.Y_test, predicted_labels, average="macro")
            print(f"Macro precision:{prec_mac}, recall:{rec_mac}, f1:{f1_mac}")
            acc = accuracy_score(self.Y_test, predicted_labels)

            distance = np.abs(self.Y_test.to_numpy()[:,0] - predicted_labels)
            # distance[distance <= 1] = 0

            print(f"Accuracy: {acc} mean distance:{np.mean(distance)}")
            if plot:
                plot_confusion_matrix(models[algo], self.X_test.ReviewMainTxt, self.Y_test)
                strategy = ""
                if overSample:
                    strategy = " with over sampling"
                format_string = ".1f"
                plt.title(f'{algo}{strategy} \nAcc:{format(acc, format_string)} Mac per:{format(prec_mac, format_string)} Mac rec:{format(rec_mac, format_string)} Mac f1:{format(f1_mac, format_string)} dist:{format(np.mean(distance), format_string)}')
                #plt.savefig(f'plots\\{algo}{strategy})_3_label.png')
                plt.show()
            else:
                cm = confusion_matrix(self.Y_test, predicted_labels)
                print(cm)

