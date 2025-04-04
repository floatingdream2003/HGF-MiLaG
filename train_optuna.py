import argparse
import torch
import himallgg
import os
import optuna

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

log = himallgg.utils.get_logger()

def objective(trial,args):
    # Set up hyperparameters to be optimized by Optuna
    args.learning_rate = trial.suggest_loguniform('learning_rate', 0.0002, 0.0004)
    # args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    args.hidden_size = trial.suggest_int('hidden_size', 100,200)
    args.drop_rate = trial.suggest_uniform('drop_rate', 0.1, 0.5)
    args.batch_size = trial.suggest_int('batch_size', 16, 32)
    args.wp = trial.suggest_int('wp', 5, 15)
    args.wf = trial.suggest_int('wf', 5, 15)


    # Build model
    log.debug("Building model...")
    model_file = "save/IEMOCAP/model.pt"
    model = himallgg.LGGCN(args).to(args.device)

    opt = himallgg.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = himallgg.Coach(trainset, devset, testset, model, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train the model
    log.info("Start training...")
    ret = coach.train()

    f1 = ret[4]

    # Save checkpoint
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_tes_f1": ret[4],
        "test_f1_when_best_dev": ret[3],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    # 如果当前试验表现最好，保存模型和试验参数
    if f1 > (objective.best_f1 if hasattr(objective, 'best_f1') else 0):
        objective.best_f1 = f1
        torch.save(checkpoint, model_file)
        with open('best_trial.txt', 'w') as f:
            f.write(f"Best trial parameters: {trial.params}\n")
            f.write(f"Best trial value: {f1}\n")

    return f1

def main(args):
    himallgg.utils.set_seed(args.seed)

    # 加载数据
    log.debug("Loading data from '%s'." % args.data)
    data = himallgg.utils.load_pkl(args.data)
    log.info("Loaded data.")

    global trainset, devset, testset, model_file
    trainset = himallgg.Dataset(data["train"], args.batch_size)
    devset = himallgg.Dataset(data["dev"], args.batch_size)
    testset = himallgg.Dataset(data["test"], args.batch_size)
    model_file = "save/IEMOCAP/model.pt"

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trail: objective(trail,args), n_trials=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")

    # Training parameters
    parser.add_argument("--from_begin", action="store_true",
                        help="Training from begin.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.0003,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.4,
                        help="Dropout rate.")

    # Model parameters
    parser.add_argument("--wp", type=int, default=10,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10,
                        help="Future context window size. Set wp to -1 to use all the future context.")
    parser.add_argument("--n_speakers", type=int, default=2,
                        help="Number of speakers.")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru"], help="Type of RNN cell.")
    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")

    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    # 24seed 63
    args = parser.parse_args()
    log.debug(args)

    main(args)

