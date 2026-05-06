from sklearn.ensemble import RandomForestClassifier

from scripts import config
from scripts.models.classical.common import build_arg_parser, train_classifier


MODEL_NAME = "random_forest"


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
        max_depth=config.RANDOM_FOREST_MAX_DEPTH,
        class_weight="balanced",
        random_state=config.SEED,
        n_jobs=-1,
    )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_classifier(build_model(), MODEL_NAME, args.features)
