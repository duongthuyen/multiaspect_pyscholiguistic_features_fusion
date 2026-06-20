from scripts import config


MODEL_NAME = "xgboost"


def build_model(seed: int = config.SEED):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for this classifier. Install requirements.txt first."
        ) from exc

    return XGBClassifier(
        n_estimators=config.XGBOOST_N_ESTIMATORS,
        max_depth=config.XGBOOST_MAX_DEPTH,
        learning_rate=config.XGBOOST_LEARNING_RATE,
        objective="multi:softprob",
        num_class=config.NUM_LABELS,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )


if __name__ == "__main__":
    from scripts.training.traditional import build_arg_parser, train_classifier
    args = build_arg_parser().parse_args()
    train_classifier(build_model(args.seed), MODEL_NAME, args.features,
                     svd_components=args.svd, seed=args.seed)
