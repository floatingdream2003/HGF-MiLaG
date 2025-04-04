import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import himallgg

log = himallgg.utils.get_logger()


class Prediction:

    def __init__(self, testset, model, args):
        self.testset = testset
        self.model = model
        self.args = args
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_tes_f1 = ckpt["best_tes_f1"]
        self.test_f1_when_best_dev = ckpt['test_f1_when_best_dev']
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def evaluate(self):
        dataset = self.testset

        self.model.eval()
        with torch.no_grad():
            golds = []#原
            golds_gender = []#后加的
            # preds = []
            preds1 = []  # 用于保存第一个输出
            preds2 = []  # 用于保存第二个输出
            for idx in tqdm(range(len(dataset)), desc="test"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                # print("gold",type(golds))
                golds_gender.append(data["xingbie_tensor"])
                for k, v in data.items():
                    if k == 'sentence':
                        continue
                    else:
                        data[k] = v.to(self.args.device)
                # y_hat = self.model(data)
                y_hat1, y_hat2 = self.model(data)  # 加入性别后，多了一个结果值


                #print(y_hat)#测试
                # preds.append(y_hat.detach().to("cpu"))#原
                preds1.append(y_hat1.detach().to("cpu"))  # 第一个输出
                preds2.append(y_hat2.detach().to("cpu"))  # 第二个输出
                # print("preds",type(preds1))

                # 将序号、真实标签和预测标签写入文件
                # if idx==0:
                #     with open("labels_and_predictions.txt", "w") as f:
                #         f.write(f"{idx}\n, golds: {golds[0]}\n, preds: {preds1[0]}\n")
                # f.write(f"{golds}\n")

            golds = torch.cat(golds, dim=-1).numpy()
            golds_gender = torch.cat(golds_gender, dim=-1).numpy()
            # preds = torch.cat(preds, dim=-1).numpy()#原
            preds1 = torch.cat(preds1, dim=-1).numpy()
            preds2 = torch.cat(preds2, dim=-1).numpy()
            print("wp:",self.args.wp)
            print(metrics.classification_report(golds, preds1, digits=4))
            print("#####################################################")
            # print(metrics.classification_report(golds_gender, preds2, digits=4))
            f1 = metrics.f1_score(golds, preds1, average="weighted")
            f2 = metrics.f1_score(golds_gender, preds2, average="weighted")

        return f1,f2
