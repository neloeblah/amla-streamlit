import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

from joblib import load

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import pdb


class TorchImagePreprocessor(TransformerMixin):
    def __init__(self):
        # Numeric pipeline
        self.num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="mean")),
            ]
        )

        self.num_scale_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ]
        )
        # Categorical pipeline
        self.cat_pipe = Pipeline(
            [
                ("one-hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Column transformer (combines the above pipelines)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", self.num_pipe, num_cols),
                ("num_scale_pipeline", self.num_scale_pipe, num_cols_scale),
                ("cat_pipeline", self.cat_pipe, cat_cols),
            ],
            remainder="drop",
            n_jobs=-1,
        )

    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X, y=None):
        transformed = self.preprocessor.transform(X)
        # print(f'transformed shape: {transformed.shape}')
        # print(f'transformed: {transformed[0]}')

        return transformed


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_size,
        hidden_size,
        learning_rate=0.001,
        dropout_prob=0.4,
        batch_size=1028,
        l1_lambda=0.01,
        lr_decay_step=5,
        lr_decay_gamma=0.8,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, 1),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_log = []
        self.l1_reg_values = [[] for _ in range(self.input_size)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # self.scheduler = StepLR(self.optimizer,  step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)

    def fit(self, X, y):
        epochs = 20
        X = torch.from_numpy(X).float()

        # y_scaled
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(y)
        y_scaled = self.target_scaler.transform(y.values)
        y = torch.from_numpy(y_scaled).float().view(-1, 1)

        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X: {X[0]}")
        print(f"y: {y}")

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # l1 norm
                l1_reg = torch.tensor(0.0, requires_grad=True)
                for i, param in enumerate(self.model.parameters()):
                    l1_reg = l1_reg + torch.norm(param, 1)
                    self.l1_reg_values[i].append(torch.norm(param, 1).item())

                loss = loss + self.l1_lambda * l1_reg

                loss.backward()

                self.optimizer.step()

                epoch_losses.append(loss.item())

            epoch_loss_avg = np.mean(epoch_losses)
            self.loss_log.append(epoch_loss_avg)
            print(f"Epoch: {epoch+1}, Loss: {epoch_loss_avg:.6f}")

            # self.scheduler.step()

    def predict(self, X):
        X = torch.from_numpy(X).float()
        test_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        self.model.to("cpu")
        self.model.eval()
        pred = []
        with torch.no_grad():
            for X in test_loader:
                X = X.to("cpu")
                output = self.model(X)
                output_descaled = self.target_scaler.inverse_transform(output)
                pred.extend(output_descaled)
        return pred

    def plot_loss(self):
        plt.plot(self.loss_log)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss curve")
        plt.show()
        return self.loss_log

    def plot_norm(self):
        for i, l1_reg_values_feature in enumerate(self.l1_reg_values):
            plt.plot(l1_reg_values_feature, label=f"Feature {i + 1}")
        plt.title("L1 Regularization Path for Features")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Regularization Value")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

        return self.l1_reg_values
