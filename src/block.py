import os
import pickle
import numpy as np
from pymoo.core.problem import Problem
import pandas as pd

from src.utils import set_model_state

from tqdm import tqdm


def blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())

    return np.array(blocked_params)


def unblocker(codebook, orig_dims, blocked_params, verbose=False):

    unblocked_params = np.zeros(orig_dims)
    # start_time = time.time()
    block_idx = 0
    for idx, indices in tqdm(
        codebook.items(),
        desc=f"Unblocking D= {len(blocked_params)} ==> {orig_dims}",
        disable=not verbose,
    ):
        # st_in = time.time()
        unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])
        block_idx += 1
        # tot_in = time.time() - st_in
        # print(tot_in)

    # end_time = time.time() - start_time

    # print(end_time)
    return unblocked_params


class MultiObjOptimalBlockOptimzationProblem(Problem):
    def __init__(
        self,
        n_var=1,
        xl=2,
        xu=1024,
        params=None,
        model=None,
        evaluation=None,
        data_loader=None,
        test_loader=None,
        num_classes=None,
        device=None,
        res_path=None,
        hist_file_path=None,
        merge=True,
        model_f=None,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.model = model
        self.params = params
        self.orig_dims = len(params)
        self.evaluation = evaluation
        self.blocker = blocker
        self.unblocker = unblocker
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.res_path = res_path
        self.hist_file_path = hist_file_path
        self.dataframe = pd.read_csv(hist_file_path)
        self.merge = merge
        self.model_f = model_f

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.ones((n_pareto_points, 2))

    def _hist_block(self, B_max):
        bin_edges = np.linspace(np.min(self.params), np.max(self.params), B_max)
        # Split the data into bins
        binned_data = np.digitize(self.params, bin_edges) - 1

        blocks_arrs = [np.array([])] * B_max

        # for i in tqdm(range(self.orig_dims)):
        #     # if i % 1000000 == 0:
        #     #     print(i, self.orig_dims)
        #     b = binned_data[i]
        #     blocks_arrs[b] = np.concatenate([blocks_arrs[b], [i]])

        histogram_block_codebook = {}
        # histogram_block_codebook_size = {}
        nonempty_bins_i = 0

        # for i in range(B_max):
        #     if len(blocks_arrs[i]) > 0:
        #         histogram_block_codebook[nonempty_bins_i] = blocks_arrs[i].tolist()
        #         # histogram_block_codebook_size[i] = len(blocks_arrs[i])
        #         nonempty_bins_i += 1

        for i in tqdm(range(B_max), desc=f"Histogram K={B_max}"):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) > 0:
                histogram_block_codebook[nonempty_bins_i] = (b_i).tolist()
                # histogram_block_codebook_size[i] = len(b_i)
                nonempty_bins_i += 1

        return histogram_block_codebook

    def _merge_till_noimprv_scheme3(self, histogram_block_codebook, best_f, B_max):
        # scheme2: merge left and right parallel.

        hist_size = len(histogram_block_codebook)
        codebook = histogram_block_codebook.copy()
        addresses = list(codebook.keys())
        curr_size = hist_size
        left_pointer = int(np.floor(curr_size / 2))
        right_pointer = int(np.ceil(curr_size / 2))
        with tqdm(total=curr_size * 2) as pbar:
            while left_pointer > 0 and right_pointer < curr_size:
                addresses = list(codebook.keys())

                # Left
                if left_pointer > 0:
                    l1, l2 = addresses[left_pointer - 1], addresses[left_pointer]
                    left_mrg_codebook = codebook.copy()
                    left_mrg_codebook[l1] = np.concatenate(
                        (
                            left_mrg_codebook[l1],
                            left_mrg_codebook[l2],
                        )
                    ).tolist()
                    del left_mrg_codebook[l2]

                    left_merged_params = self.unblocker(
                        left_mrg_codebook,
                        self.orig_dims,
                        blocked_params=self.blocker(self.params, left_mrg_codebook),
                    )

                    model = set_model_state(
                        model=self.model, parameters=left_merged_params
                    )
                    left_f = self.evaluation(
                        model,
                        data_loader=self.data_loader,
                        num_classes=self.num_classes,
                        device=self.device,
                    )
                else:
                    left_f = 1.0

                # Right
                if right_pointer < curr_size and right_pointer != left_pointer:
                    r1, r2 = addresses[right_pointer - 1], addresses[right_pointer]
                    right_mrg_codebook = codebook.copy()
                    right_mrg_codebook[r1] = np.concatenate(
                        (right_mrg_codebook[r1], right_mrg_codebook[r2])
                    ).tolist()
                    del right_mrg_codebook[r2]

                    right_merged_params = self.unblocker(
                        right_mrg_codebook,
                        self.orig_dims,
                        blocked_params=self.blocker(self.params, right_mrg_codebook),
                    )

                    model = set_model_state(
                        model=self.model, parameters=right_merged_params
                    )
                    right_f = self.evaluation(
                        model,
                        data_loader=self.data_loader,
                        num_classes=self.num_classes,
                        device=self.device,
                    )
                else:
                    right_f = 1.0

                if right_f < best_f and left_f < best_f:
                    right_mrg_codebook[l1] = left_mrg_codebook[addresses[l1]]
                    if l2 < len(right_mrg_codebook) and l2 != r2:
                        del right_mrg_codebook[l2]
                    codebook = right_mrg_codebook.copy()
                    right_pointer -= 1
                elif left_f < best_f:
                    # best_f = left_f
                    codebook = left_mrg_codebook.copy()
                    right_pointer += 1
                elif right_f < best_f:
                    # best_f = right_f
                    codebook = right_mrg_codebook.copy()
                    left_pointer -= 1
                else:
                    left_pointer -= 1
                    right_pointer += 1

                    pbar.update(1)
                curr_size = len(codebook)
                print(
                    f"B_max:{B_max}, current size:{curr_size}"
                    f" | curr best f1: {best_f:.9f}, L f1: {left_f:.9f}, R f1: {right_f:.9f}",
                )
        return codebook

    def _merge_till_noimprv_scheme2(self, histogram_block_codebook, best_f, B_max):
        # scheme2: merge left and right parallel.

        hist_size = len(histogram_block_codebook)
        codebook = histogram_block_codebook.copy()
        # addresses = list(codebook.keys())
        curr_size = hist_size
        left_pointer = 1
        right_pointer = curr_size - 1
        with tqdm(total=curr_size) as pbar:
            while left_pointer < curr_size and right_pointer > 0:
                addresses = list(codebook.keys())
                # Left
                l1, l2 = addresses[left_pointer - 1], addresses[left_pointer]
                left_mrg_codebook = codebook.copy()
                left_mrg_codebook[l1] = np.concatenate(
                    (
                        left_mrg_codebook[l1],
                        left_mrg_codebook[l2],
                    )
                ).tolist()
                del left_mrg_codebook[l2]

                left_merged_params = self.unblocker(
                    left_mrg_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, left_mrg_codebook),
                )

                model = set_model_state(model=self.model, parameters=left_merged_params)
                left_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )

                # Right
                r1, r2 = addresses[right_pointer - 1], addresses[right_pointer]
                right_mrg_codebook = codebook.copy()
                right_mrg_codebook[r1] = np.concatenate(
                    (right_mrg_codebook[r1], right_mrg_codebook[r2])
                ).tolist()
                del right_mrg_codebook[r2]

                right_merged_params = self.unblocker(
                    right_mrg_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, right_mrg_codebook),
                )

                model = set_model_state(
                    model=self.model, parameters=right_merged_params
                )
                right_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                if right_f < best_f and left_f < best_f:
                    right_mrg_codebook[l1] = left_mrg_codebook[l1]
                    if l2 < len(right_mrg_codebook) and l2 != r2:
                        del right_mrg_codebook[l2]
                    codebook = right_mrg_codebook.copy()
                    curr_size -= 2
                    right_pointer -= 1
                elif left_f < best_f:
                    # best_f = left_f
                    codebook = left_mrg_codebook.copy()
                    curr_size -= 1
                    # pbar.update(1)
                elif right_f < best_f:
                    # best_f = right_f
                    codebook = right_mrg_codebook.copy()
                    curr_size -= 1
                    left_pointer += 1
                    # pbar.update(1)
                else:
                    left_pointer += 1
                    pbar.update(1)

                right_pointer -= 1

                # print(
                #     f"B_max:{B_max}, current size:{curr_size}"
                #     f" | curr best f1: {best_f:.9f}, L f1: {left_f:.9f}, R f1: {right_f:.9f}",
                # )
        return codebook

    def _merge_till_noimprv_scheme1(self, histogram_block_codebook, best_f, B_max):
        # scheme1: merge from left to right

        hist_size = len(histogram_block_codebook)
        codebook = histogram_block_codebook.copy()
        curr_size = hist_size
        i = 1
        left_pointer = 1
        with tqdm(total=curr_size) as pbar:
            while i < curr_size - 1:
                addresses = list(codebook.keys())
                l1, l2 = addresses[left_pointer - 1], addresses[left_pointer]
                # Left
                left_mrg_codebook = codebook.copy()
                left_mrg_codebook[l1] = np.concatenate(
                    (
                        left_mrg_codebook[l1],
                        left_mrg_codebook[l2],
                    )
                ).tolist()
                del left_mrg_codebook[l2]

                left_merged_params = self.unblocker(
                    left_mrg_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, left_mrg_codebook),
                )

                model = set_model_state(model=self.model, parameters=left_merged_params)
                left_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                # Right
                l1, l2 = addresses[left_pointer], addresses[left_pointer + 1]
                # Left
                right_mrg_codebook = codebook.copy()
                right_mrg_codebook[l1] = np.concatenate(
                    (
                        right_mrg_codebook[l1],
                        right_mrg_codebook[l2],
                    )
                ).tolist()
                del right_mrg_codebook[l2]

                right_merged_params = self.unblocker(
                    right_mrg_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, right_mrg_codebook),
                )

                model = set_model_state(
                    model=self.model, parameters=right_merged_params
                )
                right_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )

                print(
                    f"B_max:{B_max}, curr size:{len(codebook)}, dim:{i}/{curr_size}",
                    f" | curr best f1: {best_f:.9f}, L f1: {left_f:.9f}, R f1: {right_f:.9f}",
                )

                argmax_f = np.argmin([left_f, right_f, best_f])
                if argmax_f == 0:
                    codebook = left_mrg_codebook
                    # best_f = left_f
                    curr_size -= 1
                    # i -= 1
                elif argmax_f == 1:
                    codebook = right_mrg_codebook
                    # best_f = right_f
                    curr_size -= 1
                    # i -= 1
                else:
                    left_pointer += 1
                    i += 1
                    pbar.update(1)
        return codebook

    def _evaluate(self, X, out, *args, **kwargs):

        NP = X.shape[0]
        f1 = np.zeros(NP)
        f2 = np.zeros(NP)
        for si, B_max in enumerate(X):
            B_max = int(B_max[0])
            xopt_codebook, xopt_f = None, None
            if self.merge == True:
                xhist_codebook = self._hist_block(B_max)
                un_params = self.unblocker(
                    xhist_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, xhist_codebook),
                )
                model = set_model_state(model=self.model, parameters=un_params)

                xhist_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                xhist_test_f = self.evaluation(
                    model,
                    data_loader=self.test_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                    mode="test",
                )

                x_path = f"{self.res_path}/codebooks/merged_codebook_bmax_{B_max}.pkl"
                if os.path.exists(x_path):
                    with open(
                        x_path,
                        "rb",
                    ) as f:
                        xopt_codebook = pickle.load(f)
                else:

                    xopt_codebook = self._merge_till_noimprv_scheme1(
                        xhist_codebook, self.model_f, B_max
                    )

                un_params = self.unblocker(
                    xopt_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, xopt_codebook),
                )
                model = set_model_state(model=self.model, parameters=un_params)
                xopt_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                xopt_test_f = self.evaluation(
                    model,
                    data_loader=self.test_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                    mode="test",
                )

                x_path = f"{self.res_path}/codebooks/merged_codebook_bmax_{B_max}.pkl"
                with open(
                    x_path,
                    "wb",
                ) as f:
                    pickle.dump(xopt_codebook, f)

                B_opt = len(xopt_codebook)
                f1[si] = xopt_f
                f2[si] = B_opt

                new_row = pd.DataFrame(
                    {
                        "B_max": [B_max],
                        "B_max_f1": [xhist_f],
                        "B_max_test_f1": [xhist_test_f],
                        "B_opt": [B_opt],
                        "B_opt_f1": [xopt_f],
                        "B_opt_test_f1": [xopt_test_f],
                    }
                )
                self.dataframe = pd.concat([self.dataframe, new_row])
                self.dataframe.to_csv(self.hist_file_path, index=False)
            else:

                if sum(self.dataframe["B_max"] == B_max) > 0:
                    results = self.dataframe.loc[self.dataframe["B_max"] == B_max]

                    f1[si] = float(results["B_opt_f1"].iloc[0])

                    f2[si] = float(results["B_opt"].iloc[0])
                    continue

                xhist_codebook = None
                x_path = f"{self.res_path}/codebooks/codebook_bmax_{B_max}.pkl"
                if os.path.exists(x_path):
                    with open(
                        x_path,
                        "rb",
                    ) as f:
                        xhist_codebook = pickle.load(f)
                else:
                    xhist_codebook = self._hist_block(B_max)

                un_params = self.unblocker(
                    xhist_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, xhist_codebook),
                )
                model = set_model_state(model=self.model, parameters=un_params)
                xhist_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )

                xhist_test_f = self.evaluation(
                    model,
                    data_loader=self.test_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                    mode="test",
                )

                B_opt = len(xhist_codebook)
                f1[si] = xhist_f
                f2[si] = B_opt

                new_row = pd.DataFrame(
                    {
                        "B_max": [B_max],
                        "B_max_f1": [xhist_f],
                        "B_max_test_f1": [xhist_test_f],
                        "B_opt": [B_opt],
                        "B_opt_f1": [xhist_f],
                        "B_opt_test_f1": [xhist_test_f],
                    }
                )
                self.dataframe = pd.concat([self.dataframe, new_row])
                self.dataframe.to_csv(self.hist_file_path, index=False)

        out["F"] = np.column_stack([f1, f2])
        return out
