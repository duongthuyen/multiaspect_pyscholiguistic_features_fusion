from sklearn.linear_model import LogisticRegression

from scripts import config
from scripts.models.classical.common import build_arg_parser, train_classifier


MODEL_NAME = "logistic_regression"


def build_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
        class_weight="balanced",
        random_state=config.SEED,
        n_jobs=-1,
    )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_classifier(build_model(), MODEL_NAME, args.features)
