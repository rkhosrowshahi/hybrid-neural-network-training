import time
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
)

from torcheval.metrics.functional import multiclass_f1_score, topk_multilabel_accuracy

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])


from pymoo.core.problem import Problem
from pymoo.core.callback import Callback

from tqdm import tqdm


class GFOProblem(Problem):
    def __init__(
        self,
        n_var=None,
        model=None,
        dataset=None,
        batch_size=1024,
        num_classes=10,
        block=False,
        codebook=None,
        orig_dims=None,
        set_model_state=None,
        device=None,
        criterion=None,
        test_loader=None,
        train_loader=None,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=-5.0,
            xu=5.0,
            vtype=float,
        )
        self.model = model
        self.batch_size = batch_size
        self.block = block  # Enable / disable block
        self.dataset = dataset
        self.num_classes = num_classes
        self.test_loader = test_loader
        self.set_model_state = set_model_state
        self.device = device
        self.codebook = codebook
        self.orig_dims = orig_dims
        self.criterion = criterion
        if criterion is None:
            self.fitness_func = self.f1score_func
        elif criterion == "crossentropy":
            self.fitness_func = self.crossentropy_func
        elif criterion == "f1":
            self.fitness_func = self.f1score_func
        elif criterion == "top1":
            self.fitness_func = self.top1_func

        if train_loader is None:
            self.data_loader = self.data_sampler()
        else:
            self.data_loader = train_loader

        self.fitness = None

    def data_sampler(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=2,
            # pin_memory=True,
        )

    def update_unblocked_params(self, block_idx, indices, blocked_params):
        self.unblocked_params[indices] = np.full(
            len(indices), blocked_params[block_idx]
        )

    def unblocker(self, blocked_params, verbose=False):
        unblocked_params = np.zeros(self.orig_dims)
        block_idx = 0
        for idx, indices in tqdm(
            self.codebook.items(),
            desc=f"Unblocking D= {len(blocked_params)} ==> {self.orig_dims}",
            disable=not verbose,
        ):
            unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])
            block_idx += 1
        return unblocked_params

    def crossentropy_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        with torch.no_grad():
            data, target = next(iter(data_loader))
            data, target = data.to(device), target.to(device)
            output = model(data)
            fitness += torch.nn.functional.cross_entropy(output, target).item()

        return fitness

    def f1score_func(self, model, data_loader, device):
        model.eval()
        fitness = 0
        all_preds, all_labels = (
            torch.Tensor([]).to(device),
            torch.Tensor([]).to(device),
        )
        t1 = time.time()
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, preds = torch.max(outputs, 1)

                all_preds = torch.cat((all_preds, preds))
                all_labels = torch.cat((all_labels, labels))
                # )
                # if mode == "val":
                #     break
        t2 = time.time()
        fitness = f1_score(
            y_true=all_labels.cpu().detach(),
            y_pred=all_preds.cpu().detach(),
            average="macro",
            labels=np.arange(self.num_classes),
        )
        t3 = time.time()
        return fitness

    def top1_func(self, model, data_loader, device):
        model.eval()
        fitness = 0
        all_outputs, all_labels = (
            torch.Tensor([]).to(device),
            torch.Tensor([]).to(device),
        )
        t1 = time.time()
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                all_outputs = torch.cat((all_outputs, outputs))
                all_labels = torch.cat((all_labels, labels))
                # )
                # if mode == "val":
                #     break
        t2 = time.time()
        fitness = top_k_accuracy_score(
            y_true=all_labels.cpu().detach(),
            y_score=all_outputs.cpu().detach(),
            k=1,
            labels=np.arange(self.num_classes),
        )

        return fitness

    def test_func(self, X):
        uxi = X.copy()
        if self.block:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)
        self.model.eval()

        all_outputs, all_preds, all_labels = (
            torch.Tensor([]).to(self.device),
            torch.Tensor([]).to(self.device),
            torch.Tensor([]).to(self.device),
        )
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, preds = torch.max(output, 1)
                all_outputs = torch.cat((all_outputs, output))
                all_preds = torch.cat((all_preds, preds))
                all_labels = torch.cat((all_labels, target))
        f1_f = f1_score(
            all_labels.cpu().detach(),
            all_preds.cpu().detach(),
            average="macro",
            labels=np.arange(self.num_classes),
        )
        top1_f = top_k_accuracy_score(
            all_labels.cpu().detach(),
            all_outputs.cpu().detach(),
            k=1,
            labels=np.arange(self.num_classes),
        )
        return {
            "f1": f1_f,
            "top1": top1_f,
        }

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

    def _evaluate(self, X, out, verbose=True, *args, **kwargs):
        NP = len(X)
        fout = np.zeros(NP)

        # for i in range(NP):
        for i in tqdm(
            range(NP),
            desc=f"Evaluation...",
            disable=not verbose,
        ):
            uxi = X[i]
            if self.block and len(uxi) != self.orig_dims:
                uxi = self.unblocker(uxi)

            self.set_model_state(model=self.model, parameters=uxi)

            fitness = self.fitness_func(
                model=self.model, data_loader=self.data_loader, device=self.device
            )
            fout[i] = fitness

        out["F"] = fout

    def individual_fitness(self, X):

        uxi = X
        if self.block and len(uxi) != self.orig_dims:
            uxi = self.unblocker(uxi)

        loaded_model = self.set_model_state(model=self.model, parameters=uxi)

        fitness = self.fitness_func(
            model=loaded_model, data_loader=self.data_loader, device=self.device
        )

        return fitness

    def multithread_fitness_func(self, X, idx):

        uxi = X
        if self.block and len(uxi) != self.orig_dims:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)

        self.fitness[idx] = self.fitness_func(
            model=self.model, data_loader=self.data_loader, device=self.device
        )

        # return fitness


class SOCallback(Callback):

    def __init__(
        self,
        k_FEs=10,
        problem=None,
        csv_path=None,
        plt_path=None,
        start_eval=1,
        start_iter=0,
    ) -> None:
        super().__init__()
        self.k_FEs = k_FEs
        self.csv_path = csv_path
        self.plt_path = plt_path
        self.problem = problem

        self.data["opt_F"] = []
        self.data["pop_F"] = []
        self.data["n_evals"] = []
        self.start_iter = start_iter
        self.start_eval = start_eval

    def notify(self, algorithm):
        self.data["opt_F"].append(algorithm.opt.get("F")[0][0])
        self.data["pop_F"].append(algorithm.pop.get("F"))
        self.data["n_evals"].append(algorithm.evaluator.n_eval)

        df = pd.read_csv(self.csv_path)
        # if len(df) >= 2:

        if (self.start_eval + algorithm.evaluator.n_eval) % self.k_FEs == 0:
            best_X = algorithm.opt.get("X")[0]

            NP = len(algorithm.pop)
            # algorithm.evaluator.n_eval += NP
            test_fs = algorithm.problem.test_func(best_X)
            # Define the new row as a dictionary
            new_row = {
                "n_step": self.start_iter + algorithm.n_iter,
                "n_eval": self.start_eval + algorithm.evaluator.n_eval,
                "f_best": algorithm.opt.get("F")[0][0],
                "f_avg": algorithm.pop.get("F").mean(),
                "f_std": algorithm.pop.get("F").std(),
                "test_f1_best": test_fs["f1"],
                "test_top1_best": test_fs["top1"],
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"].to_numpy(), df["f_best"].to_numpy(), label="train")
            if algorithm.problem.criterion != "crossentropy":
                plt.plot(
                    df["n_step"].to_numpy(),
                    df["test_f1_best"].to_numpy(),
                    label="test",
                )
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.title(f"{algorithm.__class__.__name__}, {algorithm.problem.criterion}")
            plt.legend()
            plt.grid()
            plt.savefig(self.plt_path)
            plt.close()

    def scipy_func(self, intermediate_result):

        if intermediate_result.nit % self.k_steps == 0:
            best_X, best_F = intermediate_result.x, intermediate_result.fun
            df = pd.read_csv(self.csv_path)

            test_fs = self.problem.test_func(best_X)
            # Define the new row as a dictionary
            new_row = {
                "n_step": intermediate_result.nit,
                "n_eval": intermediate_result.nfev,
                "f_best": best_F,
                "f_avg": intermediate_result.population_energies.mean(),
                "f_std": intermediate_result.population_energies.std(),
                "test_f1_best": test_fs["f1"],
                "test_top1_best": test_fs["top1"],
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"].to_numpy(), df["f_best"].to_numpy(), label="train")
            if self.problem.criterion != "crossentropy":
                plt.plot(
                    df["n_step"].to_numpy(),
                    df["test_f1_best"].to_numpy(),
                    label="test",
                )
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.legend()
            plt.grid()
            plt.savefig(self.plt_path)
            plt.close()

    def general_caller(
        self,
        niter,
        neval,
        opt_X,
        opt_F,
        pop_F,
    ):
        if niter % 50 == 0:
            best_X, best_F = opt_X, opt_F
            df = pd.read_csv(self.csv_path)

            test_fs = self.problem.test_func(best_X)
            # Define the new row as a dictionary
            new_row = {
                "n_step": self.start_iter + niter,
                "n_eval": self.start_eval + neval,
                "f_best": best_F,
                "f_avg": pop_F.mean(),
                "f_std": pop_F.std(),
                "test_f1_best": test_fs["f1"],
                "test_top1_best": test_fs["top1"],
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            df = df.iloc[1:]

            plt.plot(df["n_eval"].to_numpy(), df["f_best"].to_numpy(), label="best")
            plt.plot(df["n_eval"].to_numpy(), df["f_avg"].to_numpy(), label="average")
            # plt.plot(df["n_eval"].to_numpy(), df["f_avg"].to_numpy(), label="average")
            if self.problem.criterion != "crossentropy":
                plt.plot(
                    df["n_eval"].to_numpy(),
                    df["test_f1_best"].to_numpy(),
                    label="best (test)",
                )
            plt.xlabel("FE")
            plt.ylabel("F1 score")
            plt.legend()
            plt.grid()
            plt.savefig(self.plt_path)
            plt.close()
