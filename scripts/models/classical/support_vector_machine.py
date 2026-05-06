from sklearn.svm import SVC

from scripts import config
from scripts.models.classical.common import build_arg_parser, train_classifier


MODEL_NAME = "support_vector_machine"


def build_model() -> SVC:
    return SVC(
        C=config.SVM_C,
        kernel=config.SVM_KERNEL,
        class_weight="balanced",
        random_state=config.SEED,
    )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_classifier(build_model(), MODEL_NAME, args.features)
