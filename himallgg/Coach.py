import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import himallgg
import pandas as pd
from sklearn.metrics import accuracy_score

log = himallgg.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.xingbie_to_idx = {'M':0, 'F':1}
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None
        self.best_gender_accuracies = []  # 用于存储性别分类的准确度
        self.best_emotion_accuracies = []  # 用于存储情感分类的准确度



    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_tes_f1 = ckpt["best_tes_f1"]
        self.test_f1_when_best_dev = ckpt['test_f1_when_best_dev']
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_tes_f1, best_epoch, best_state = self.best_dev_f1, self.best_tes_f1, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)

            dev_f1, gender_f1,dev_preds = self.evaluate()
            log.info("[Dev set] [Emotion f1 {:.4f}] [Gender f1 {:.4f}]".format(dev_f1, gender_f1))
            test_f1, test_gender_f1,test_preds = self.evaluate(test=True)
            log.info("[Test set] [Emotion f1 {:.4f}] [Gender f1 {:.4f}]".format(test_f1, test_gender_f1))


            # 更新最佳情感和性别准确度
            if best_tes_f1 is None or test_f1 > best_tes_f1:
                best_tes_f1 = test_f1
                test_f1_when_best_dev = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

                self.best_emotion_accuracies.append(dev_f1)
                self.best_gender_accuracies.append(gender_f1)

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, dev_gender_f1,dev_preds = self.evaluate()
        log.info("[Dev set] [Emotion f1 {:.4f}] [Gender f1 {:.4f}]".format(dev_f1, dev_gender_f1))


        test_f1, test_gender_f1,test_preds = self.evaluate(test=True)
        log.info("[Test set] [Emotion f1 {:.4f}] [Gender f1 {:.4f}]".format(test_f1, test_gender_f1))




        # 保存准确度到CSV文件
        self.save_test_accuracies()

        return best_dev_f1, best_epoch, best_state, test_f1_when_best_dev, best_tes_f1, test_preds

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        epoch_gender_loss = 0
        epoch_emotion_loss=0
        self.model.train()
        # for idx in tqdm(np.random.permutation(len(self.trainset)), desc="train epoch {}".format(epoch)):
        # self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                if k == 'sentence':
                    continue
                else:
                    data[k] = v.to(self.args.device)
            # nll,emotion,gender = self.model.get_loss(data)#正常的
            nll,emotion,gender = self.model.get_loss(data,epoch)#画散点图

            epoch_loss += nll.item()
            epoch_emotion_loss += emotion.item()
            epoch_gender_loss += gender.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Total Loss: %f] [Emotion Loss: %f]  [Gender Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, epoch_emotion_loss, epoch_gender_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset

        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []

            golds_gender = []  # 记录性别标签
            preds_gender = []  # 记录性别预测

            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                golds_gender.append(data["xingbie_tensor"])  # 添加性别标签
                for k, v in data.items():
                    if k == 'sentence':
                        continue
                    else:
                        data[k] = v.to(self.args.device)
                # y_hat = self.model(data)
                #################################################################################
                #决策动态
                y_hat, y_hat_gender = self.model(data)

                preds.append(y_hat.detach().to("cpu"))
                preds_gender.append(y_hat_gender.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            golds_gender = torch.cat(golds_gender, dim=-1).numpy()
            preds_gender = torch.cat(preds_gender, dim=-1).numpy()

            print(metrics.classification_report(golds, preds, digits=4))
            # print(metrics.classification_report(golds_gender, preds_gender, digits=4))  # 性别分类报告
            f1 = metrics.f1_score(golds, preds, average="weighted")
            f1_gender = metrics.f1_score(golds_gender, preds_gender, average="weighted")

            # 计算情感分类任务的准确率
            accuracy_emotion = accuracy_score(golds, preds)

        return f1,f1_gender,accuracy_emotion

    def save_test_accuracies(self):
        # 创建DataFrame并保存为CSV
        df_emotion = pd.DataFrame(self.best_emotion_accuracies, columns=["最佳情感测试准确度"])
        df_gender = pd.DataFrame(self.best_gender_accuracies, columns=["最佳性别测试准确度"])

        df_emotion.to_csv("./best_emotion.csv", index=False)
        df_gender.to_csv("./best_gender.csv", index=False)

        log.info("最佳测试准确度已保存到 best_emotion.csv 和 best_gender.csv")