from sklearn.svm import SVC

from scripts import config


MODEL_NAME = "support_vector_machine"


def build_model(seed: int = config.SEED) -> SVC:
    return SVC(
        C=config.SVM_C,
        kernel=config.SVM_KERNEL,
        class_weight="balanced",
        random_state=seed,
    )


if __name__ == "__main__":
    from scripts.training.traditional import build_arg_parser, train_classifier
    args = build_arg_parser().parse_args()
    train_classifier(build_model(args.seed), MODEL_NAME, args.features,
                     svd_components=args.svd, seed=args.seed)
