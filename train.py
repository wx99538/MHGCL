from dataclasses import dataclass, field
from typing import Tuple, List
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from DHLMCLF import DHLMCLF
import ModelEvaluate
import numpy as np
import time

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class EarlyStopping:
    """早停机制类"""
    patience: int = 60
    delta: float = 0.0001
    best_score: float = None
    counter: int = 0
    early_stop: bool = False

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

@dataclass
class TrainingConfig:
    n_splits: int = 5
    n_repeats: int = 5
    epochs: int = 2000
    learning_rate: float = 1e-4
    dropout_rate: float = 0.5
    in_dims: List[int] = field(default_factory=lambda: [200, 300, 400, 500, 600])
    k_range: range = range(2, 11)  # k1, k2, k3 的取值范围
    eval_interval: int = 20


def load_and_preprocess_data(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        clinical = pd.read_csv(f"{file_path}/cln.csv").iloc[1:, 1:]
        cnv = pd.read_csv(f"{file_path}/cnv.csv").iloc[1:, 1:]
        expression = pd.read_csv(f"{file_path}/exp.csv").iloc[1:, 1:]
        labels = pd.read_csv(f"{file_path}/y.csv").iloc[1:, 1:]

        scaler = StandardScaler()
        clinical_norm = torch.FloatTensor(scaler.fit_transform(clinical.values)).to(device)
        cnv_norm = torch.FloatTensor(scaler.fit_transform(cnv.values)).to(device)
        expression_norm = torch.FloatTensor(scaler.fit_transform(expression.values)).to(device)
        labels = torch.LongTensor(labels.values).flatten().to(device)

        return clinical_norm, cnv_norm, expression_norm, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def generate_cv_splits(*features: torch.Tensor, labels: torch.Tensor, n_splits: int = 5, n_repeats: int = 5) -> List[Tuple]:
    results = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        for train_idx, test_idx in skf.split(features[0].cpu().numpy(), labels.cpu().numpy()):
            train_data = [f[train_idx] for f in features]
            test_data = [f[test_idx] for f in features]
            full_data = [torch.cat([train, test]) for train, test in zip(train_data, test_data)]
            results.append((*train_data, labels[train_idx], *full_data, labels[test_idx]))
    return results


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                labels: torch.Tensor) -> float:
    model.train()
    optimizer.zero_grad()
    _, loss = model(*train_data, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(epoch,
             model: torch.nn.Module,
             test_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
             labels: torch.Tensor,
             eval_func=ModelEvaluate.get_result) -> Tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        # star_time = time.time()
        outputs, loss = model(*test_data, None, testing=True)
        # end_time = time.time()
        # print((end_time - star_time) * 1000)
        outputs = outputs.detach().cpu()[-len(labels):]
        labels = labels.detach().cpu()
        result = eval_func(outputs, labels)
        return loss, result


def run_training_pipeline(file_path: str, config: TrainingConfig) -> None:
    try:
        print("Loading data...")
        clinical, cnv, expression, labels = load_and_preprocess_data(file_path)

        print("Generating CV splits...")
        cv_splits = generate_cv_splits(clinical, cnv, expression, labels=labels,
                                       n_splits=config.n_splits, n_repeats=config.n_repeats)

        print("Starting training...")
        for split_idx, (x1_train, x2_train, x3_train, y_train,
                        x1_full, x2_full, x3_full, y_test) in enumerate(cv_splits, 1):
            if split_idx in [1,2,3,4,5][:]:
                train_data = (x1_train, x2_train, x3_train)
                test_data = (x1_full, x2_full, x3_full)

                # 保存原始数据用于可视化
                # test_ = torch.concatenate([x1_full, x2_full, x3_full], dim=1)
                # pd.DataFrame(test_[-len(y_test):].detach().cpu()).to_csv(r'D:\第二篇论文\mymodel\可视化\LUSC\raw.csv')
                # pd.DataFrame(y_test.detach().cpu()).to_csv(r'D:\第二篇论文\mymodel\可视化\LUSC\y_.csv')

                for in_dim in [200, 300, 400, 500, 600][:]:
                    # 独立遍历k1, k2, k3
                    for k1 in range(5, 7, 2):
                        for k2 in range(2, 11, 2):
                            for k3 in range(2, 11, 2):
                                print(f"\nTraining with in_dim={in_dim}, k1={k1}, k2={k2}, k3={k3}")
                                result = {"指标1":[], "指标2":[], "指标3":[]}
                                model = DHLMCLF(
                                    feature_dim=[x.shape[1] for x in train_data],
                                    in_dim=[in_dim, in_dim, in_dim],
                                    class_num=int(y_train.max()) + 1,
                                    k1=k1, k2=k2, k3=k3,  # 三个k值独立变化
                                    dropout=config.dropout_rate
                                ).to(device)

                                optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
                                early_stopping = EarlyStopping()
                                star_time = time.time()
                                for epoch in range(1, config.epochs + 1):
                                    train_loss = train_epoch(model, optimizer, train_data, y_train)
                                    if epoch % config.eval_interval == 0:
                                        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}", end=' ')
                                        val_loss, _ = evaluate(epoch, model, test_data=test_data, labels=y_test)
                                        result["指标1"].append(_[0])
                                        result["指标2"].append(_[1])
                                        result["指标3"].append(_[2])
                                    early_stopping(train_loss)
                                    if early_stopping.early_stop:
                                        print("Early stopping triggered, epoch is {}".format(epoch))
                                        break
                                end_time = time.time()
                                print((end_time-star_time)*1000)
                                print("Final evaluation:")
                                # star_time = time.time()
                                val_loss, _ = evaluate(config.epochs, model, test_data=test_data, labels=y_test)
                                # end_time = time.time()
                                total_params = sum(p.numel() for p in model.parameters())
                                print(f"总参数数量: {total_params:,}")
                                result["指标1"].append(_[0])
                                result["指标2"].append(_[1])
                                result["指标3"].append(_[2])
                                # pd.DataFrame(result).to_csv(r'D:\第二篇论文\收敛性\KIPAN.csv'.format(split_idx, in_dim, k1, k2, k3))
                                print("------------------")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    config = TrainingConfig()
    run_training_pipeline(
        file_path=r"D:\第二篇论文\数据集\数据\KIPAN\处理后",
        config=config
    )