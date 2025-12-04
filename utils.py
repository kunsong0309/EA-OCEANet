import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, f1_score, auc, \
    roc_auc_score, precision_recall_curve, balanced_accuracy_score, \
    recall_score, precision_score, cohen_kappa_score, \
    multilabel_confusion_matrix


class Trainer():
    def __init__(self, model, max_epoch=100, batch_size=64,
                 learning_rate=1e-3, weight_decay=0.01, patience=10,
                 epoch_check_loss=1, pretrained_dict=None):
        self.device = torch.accelerator.current_accelerator().type \
            if torch.accelerator.is_available() else "cpu"
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.epoch_check_loss = epoch_check_loss

        self.initialize_model(model)
        if pretrained_dict is not None:
            self.model.load_state_dict(torch.load(pretrained_dict))

    def initialize_model(self, model):
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=self.weight_decay,
                                           )

    def load_training_data(self, data, **data_args):
        self.train_data = data.get_dataloader('train',
                                              batch_size=self.batch_size,
                                              **data_args)
        self.val_data = data.get_dataloader('validation', **data_args)
        self.num_batch = len(self.train_data)
        self.check_loss = self.num_batch // self.epoch_check_loss

    def load_test_data(self, data, **data_args):
        self.test_data = data.get_dataloader('test', **data_args)

    def load_batch(self, batch):
        X, y = batch
        return X.to(self.device), y.to(self.device)

    def train_step(self, batch):
        self.model.train()
        X, y = self.load_batch(batch)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            pred = []
            ys = []
            for batch in self.val_data:
                X, y = self.load_batch(batch)
                pred.append(self.model(X))
                ys.append(y)
            pred = torch.cat(pred, dim=0)
            y = torch.cat(ys, dim=0)
            loss = self.loss_fn(pred, y).item()

            pred = nn.functional.softmax(pred, dim=-1).cpu()
            y = y.cpu()
            if pred.shape[-1] == 2:
                metrics = roc_auc_score(y, pred[:, 1])
            else:
                metrics = cohen_kappa_score(y, pred.argmax(-1))
        return loss, metrics

    def train_epoch(self):
        print(
            f'Epoch {self.current_epoch + 1:>3d}/{self.max_epoch:d} ----------->')
        size = len(self.train_data)
        for step, batch in enumerate(self.train_data):
            loss = self.train_step(batch)
            if (step % self.check_loss == 0) or (step == self.num_batch - 1):
                print(
                    f'Training: [{step + 1:>4d}/{size:>4d}]  loss: {loss.item():>5f}')
        self.training_log['train_loss'][self.current_epoch] = loss.item()

        loss, metrics = self.validation()
        lr = self.learning_rate
        self.training_log['val_loss'][self.current_epoch] = loss
        self.training_log['val_metrics'][self.current_epoch] = metrics
        self.training_log['learning_rate'][self.current_epoch] = lr
        print(
            f'Validation: {metrics:>5f}, loss: {loss:>5f}, lr: {lr:g}')

        self.current_epoch += 1
        if metrics > self.best_metrics:
            self.best_epoch = self.current_epoch
            self.best_loss = loss
            self.best_metrics = metrics
            self.best_model = self.model.state_dict()
            # torch.save(self.model.state_dict(), self.save_path)
        elif self.current_epoch - self.best_epoch >= self.patience:
            self.early_stop = True

    def train(self):
        self.training_log = dict(
            learning_rate=(np.zeros((self.max_epoch, )) * np.nan).tolist(),
            train_loss=(np.zeros((self.max_epoch, )) * np.nan).tolist(),
            val_loss=(np.zeros((self.max_epoch, )) * np.nan).tolist(),
            val_metrics=(np.zeros((self.max_epoch, )) * np.nan).tolist(),
        )
        self.best_epoch = 0
        self.current_epoch = 0
        self.best_loss = 1.
        self.best_metrics = 0.
        self.best_model = self.model.state_dict()
        self.early_stop = False

        for _ in range(self.max_epoch):
            self.train_epoch()
            if self.early_stop:
                break
        self.model.load_state_dict(self.best_model)
        print(
            f'Final metrics: {self.best_metrics:>5f}, loss: {self.best_loss:>5f}')

    def evaluate(self):
        if not hasattr(self, 'test_data'):
            return None

        self.model.eval()
        with torch.no_grad():
            score = []
            y = []
            for batch in self.test_data:
                X, y_ = self.load_batch(batch)
                score.append(nn.functional.softmax(
                    self.model(X), dim=-1).cpu())
                y.append(y_.cpu())
            score = torch.cat(score, dim=0)
            y = torch.cat(y, dim=0)
            pred = score.argmax(-1)

            pr_curve = precision_recall_curve(y > 0, 1 - score[:, 0])
            metrics = dict(
                f1_score=f1_score(y, pred, average=None,
                                  zero_division=0.).tolist(),
                auroc=roc_auc_score(y > 0, 1 - score[:, 0]),
                aupr=auc(pr_curve[1], pr_curve[0]),
                bacc=balanced_accuracy_score(y, pred),
                precision=precision_score(
                    y, pred, average=None, zero_division=0.).tolist(),
                recall=recall_score(y, pred, average=None,
                                    zero_division=0.).tolist(),
                specificity=specificity_score(y, pred).tolist(),
                kappa=cohen_kappa_score(y, pred),
                confusion_matrix=confusion_matrix(y, pred).tolist(),
                scores=score.numpy().tolist(),
            )
        return metrics

    def visualize(self):
        self.model.eval()
        with torch.no_grad():
            feat1 = []
            feat2 = []
            wsa = []
            wta = []
            for batch in self.test_data:
                X, _ = self.load_batch(batch)
                x1, x2, w1, w2 = map(lambda out: out.cpu(),
                                     self.model.get_features(X))
                feat1.append(x1)
                feat2.append(x2)
                wsa.append(w1)
                wta.append(w2)
            feat1 = torch.cat(feat1, dim=0)
            feat2 = torch.cat(feat2, dim=0)
            wsa = torch.cat(wsa, dim=0)
            wta = torch.cat(wta, dim=0)
        return map(lambda out: out.numpy(), (feat1, feat2, wsa, wta))


class DataModule():
    def __init__(
        self,
        data_path,
        sub_list,
        normalize=True,
        shuffle=True,
        kfold=5,
        percent=1.,
        seed=42,
        channels=1,
        **kwargs,
    ):
        self.ds = dict()
        for sub in sub_list:
            X = np.load(data_path + sub + '_X.npy', allow_pickle=True)
            Y = np.load(data_path + sub + '_Y.npy', allow_pickle=True)

            X = X.astype(np.float32)
            if normalize:
                X /= (np.percentile(np.abs(X), 95, axis=-1)
                      [:, :, :, None] + 1e-8)
            if channels != 1:
                X = np.tile(X, (1, 1, channels, 1))

            Y = Y.astype(np.int64).flatten()

            fold = np.ones(Y.shape, dtype=np.int16) * kfold
            if shuffle:
                np.random.seed(seed)
                for cate in np.arange(np.max(Y) + 1):
                    ido = np.argwhere(Y == cate)[:, 0]
                    nsamp = len(ido)
                    ntrain = int(nsamp * percent)
                    nfold = int(np.ceil(nsamp / kfold))
                    ids = np.random.permutation(nsamp)
                    idf = np.tile(np.arange(kfold), (nfold, 1)).flatten()
                    fold[ido[ids[:ntrain]]] = idf[:ntrain]
            else:
                nsamp = Y.shape[0]
                nfold = int(np.ceil(nsamp / kfold))
                idf = np.tile(np.arange(kfold), (nfold, 1)).T.flatten()
                fold[:] = idf[:nsamp]

            self.ds[sub] = []
            for k in range(kfold):
                self.ds[sub].append(
                    [(data, target) for data, target in zip(X[fold == k, :], Y[fold == k])])

    def get_dataloader(self, part, gid=None, sid=None, batch_size=None, seed=7468, mode='total'):
        np.random.seed(seed)
        data = []
        if part == 'test':
            for key, value in self.ds.items():
                if key in gid:
                    for v in value:
                        data += v
        elif part == 'validation':
            for key, value in self.ds.items():
                if key not in gid:
                    for k, v in enumerate(value):
                        if k in sid:
                            data += v
        elif part == 'train':
            for key, value in self.ds.items():
                if key not in gid:
                    for k, v in enumerate(value):
                        if k not in sid:
                            data += v
        else:
            return None

        if part == 'train':
            y = np.array([y for _, y in data])
            _, counts = np.unique(y, return_counts=True)
            w = 100 / counts[y]
            if mode == 'total':
                tol = len(y)
            else:
                tol = int(counts[mode] * len(counts))
            sampler = WeightedRandomSampler(w, tol, replacement=True,
                                            generator=torch.Generator().manual_seed(seed))
        else:
            batch_size = min(len(data), 64)
            sampler = None

        return DataLoader(data, batch_size=batch_size, sampler=sampler)


def specificity_score(ytrue, ypred):
    cm = multilabel_confusion_matrix(ytrue, ypred)
    return cm[:, 0, 0] / (cm[:, 0, 1] + cm[:, 0, 0] + 1e-8)
