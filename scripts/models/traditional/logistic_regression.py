from sklearn.linear_model import LogisticRegression

from scripts import config


MODEL_NAME = "logistic_regression"


def build_model(seed: int = config.SEED) -> LogisticRegression:
    return LogisticRegression(
        max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


if __name__ == "__main__":
    from scripts.training.traditional import build_arg_parser, train_classifier
    args = build_arg_parser().parse_args()
    train_classifier(build_model(args.seed), MODEL_NAME, args.features,
                     svd_components=args.svd, seed=args.seed)
