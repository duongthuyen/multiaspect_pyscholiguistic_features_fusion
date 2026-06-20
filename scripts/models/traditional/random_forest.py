from sklearn.ensemble import RandomForestClassifier

from scripts import config


MODEL_NAME = "random_forest"


def build_model(seed: int = config.SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
        max_depth=config.RANDOM_FOREST_MAX_DEPTH,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


if __name__ == "__main__":
    from scripts.training.traditional import build_arg_parser, train_classifier
    args = build_arg_parser().parse_args()
    train_classifier(build_model(args.seed), MODEL_NAME, args.features,
                     svd_components=args.svd, seed=args.seed)
