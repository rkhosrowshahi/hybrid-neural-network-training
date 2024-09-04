import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
from tqdm import tqdm

from src.gfo import GFOProblem, SOCallback
from src.dataloader import get_val_test_dataloader
from src.utils import (
    get_model_params,
    get_network,
    init_pop_in_block,
    set_model_state,
    set_seed,
)
from src.block import blocker, unblocker
from evosax import DE, CMA_ES, PSO, OpenES, SimpleES


def main(args):

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    # set_seed(args.seed)

    batch_size = 128
    val_loader, test_loader, num_classes = get_val_test_dataloader(
        dataset=args.dataset, batch_size=batch_size
    )

    model = get_network(args.net, args.dataset, args.model_path)
    model.to(device)
    model_params = get_model_params(model)
    D = len(model_params)

    codebook = None
    with open(f"codebooks/{args.net}_{args.dataset}_1e_codebook.pkl", "rb") as f:
        codebook = pickle.load(f)
    BD = len(codebook)
    print(BD)

    init_params_blocked = blocker(model_params, codebook)
    x0 = init_params_blocked.copy()

    rng = jax.random.PRNGKey(0)
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    problem = GFOProblem(
        n_var=BD,
        model=model,
        dataset=None,
        test_loader=test_loader,
        train_loader=val_loader,
        set_model_state=set_model_state,
        batch_size=batch_size,
        device=device,
        criterion="top1",
        block=True,
        codebook=codebook,
        orig_dims=D,
        num_classes=num_classes,
    )

    optimizer = None
    if args.solver.lower() == "de":
        optimizer = DE(popsize=100, num_dims=BD, maximize=True)
    if args.solver.lower() == "pso":
        optimizer = PSO(popsize=100, num_dims=BD, maximize=True)
    elif args.solver.lower() == "cma-es":
        optimizer = CMA_ES(
            popsize=4 + int(np.floor(3 * np.log(BD))),
            # popsize=100,
            num_dims=BD,
            sigma_init=0.1,
            maximize=True,
        )
    elif args.solver.lower() == "simple-es":
        optimizer = SimpleES(popsize=100, num_dims=BD, sigma_init=0.002, maximize=True)
    elif args.solver.lower() == "open-es":
        optimizer = OpenES(popsize=100, num_dims=BD, opt_name="adam", maximize=True)

    NP = optimizer.popsize
    es_params = optimizer.default_params
    print(np.min(x0), np.max(x0))
    # es_params = es_params.replace(clip_min=np.min(x0), clip_max=np.max(x0))

    if args.solver.lower() == "de":
        es_params = es_params.replace(
            # init_min=np.min(x0),
            # init_max=np.max(x0),
            diff_w=0.5,
            cross_over_rate=0.7,
            mutate_best_vector=True,  # Maybe using best vector in mutation helps to converge faster due to using the pretrained params in population
        )
    elif args.solver.lower() == "pso":
        es_params = es_params.replace(
            # init_min=np.min(x0),
            # init_max=np.max(x0),
            inertia_coeff=0.729844,  # w momentum of velocity
            cognitive_coeff=1.49618,  # c_1 cognitive "force" multiplier
            social_coeff=1.49618,  # c_2 social "force" multiplier
        )
    # elif args.solver.lower() == "cma-es":

    # init_pop = jax.random.uniform(rng, (NP, BD), minval=-5, maxval=5)
    init_pop = init_pop_in_block(NP, codebook, model_params)
    # init_pop = init_pop.at[0].set(jnp.array(x0))

    state = optimizer.initialize(rng, es_params)
    # state = state.replace(archive=init_pop, best_member=jnp.array(x0))

    maxFE = 3000000
    FE = 0
    iters = 0
    verbose = True

    uxi = unblocker(codebook=codebook, orig_dims=D, blocked_params=x0)
    set_model_state(model=model, parameters=uxi)
    if problem.criterion == "f1":
        best_F = problem.f1score_func(model, val_loader, device)
    elif problem.criterion == "top1":
        best_F = problem.top1_func(model, val_loader, device)
    best_x0 = x0
    print(f"Block Model F: {best_F:.6f}")

    if "es" in args.solver.lower():
        state = state.replace(mean=x0)

    csv_path = f"outs/{args.net}_{args.dataset}_{args.solver.lower()}_hist.csv"
    plt_path = f"outs/{args.net}_{args.dataset}_{args.solver.lower()}_plt.pdf"

    df = pd.DataFrame(
        {
            "n_step": [0],
            "n_eval": [1],
            "f_best": [best_F],
            "f_avg": [0],
            "f_std": [0],
            "test_f1_best": problem.f1score_func(model, test_loader, device),
            "test_top1_best": problem.top1_func(model, test_loader, device),
        }
    )
    df.to_csv(csv_path, index=False)

    callback = SOCallback(
        k_steps=50,
        csv_path=csv_path,
        plt_path=plt_path,
        start_eval=FE,
        start_iter=iters,
        problem=problem,
    )
    scale_type = None
    if args.solver.lower() == "de":
        scale_type = "mutation_rate"
    elif "es" in args.solver.lower():
        scale_type = "sigma"
    # else:
    #     scale_type = "N/A"
    print(
        f"n_step, n_eval, best F, pop F_best, pop F_mean, pop F_std, pop X_low, pop X_high, time_step",
        end=", ",
    )
    if scale_type is not None:
        print(f"{scale_type}")
    else:
        print()
    while FE <= maxFE:
        t1 = time.time()

        if args.solver.lower() == "de":
            es_params = es_params.replace(
                diff_w=np.random.uniform(low=0, high=0.1, size=1)[0]
            )

        pop_X, state = optimizer.ask(rng_gen, state, es_params)

        if FE == 0:
            if args.solver.lower() == "de" or args.solver.lower() == "pso":
                pop_X = jnp.array(init_pop)
                pop_X = pop_X.at[0].set(x0)
        pop_F = np.zeros(NP)
        for ip in tqdm(
            range(NP),
            desc=f"Evaluation...",
            disable=not verbose,
        ):
            f = problem.individual_fitness(pop_X[ip])
            pop_F[ip] = f

        state = optimizer.tell(pop_X, pop_F, state, es_params)

        t2 = time.time()

        argmax = pop_F.argmax()
        best_pop_F = pop_F[argmax]
        if best_pop_F >= best_F:
            best_F = best_pop_F
            best_x0 = pop_X[argmax]

        FE += optimizer.popsize
        iters += 1
        callback.general_caller(
            niter=iters, neval=FE, opt_X=best_x0, opt_F=best_F, pop_F=pop_F
        )
        scale = None
        if args.solver.lower() == "de":
            scale = str(round(es_params.diff_w, 6))
        elif "es" in args.solver.lower():
            scale = str(round(np.mean(state.sigma), 6))
        # else:
        #     scale = "N/A"

        print(
            f"{iters}, {FE}, {best_F:.6f}, {best_pop_F:.6f}, {pop_F.mean():.6f}, {pop_F.std():.6f}, {pop_X.min():.6f}, {pop_X.max():.6f}, {(t2-t1):.1f}",
            end=", ",
        )
        if scale is not None:
            print(f"{scale}")
        else:
            print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=False, help="network type")
    parser.add_argument(
        "--solver", type=str, required=False, help="optimizer type, e.g., DE or CMA-ES"
    )
    parser.add_argument(
        "--model_path", type=str, required=False, help="model checkpoint path"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, help="model checkpoint path"
    )
    parser.add_argument(
        "--gpu", type=str, default="cuda:0", help="use cuda:[number1], cuda:[number2]"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="use cifar10, cifar100 or imagenet",
    )
    parser.add_argument("--b", type=int, default=128, help="batch size for dataloader")
    parser.add_argument(
        "--seed", type=int, default=1, help="seed value for random values"
    )
    parser.add_argument(
        "--lb", type=int, default=2, help="lower bound in population initilization"
    )
    parser.add_argument(
        "--ub", type=int, default=1000, help="upper bound in population initilization"
    )

    args = parser.parse_args()

    main(args)

# FOR DEBUGGING jax.debug.print("{}", best_archive_fitness[0])
