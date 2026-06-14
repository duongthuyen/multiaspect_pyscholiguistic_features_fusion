from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.features.base import FeatureExtractorBase
from scripts.features.semantic.mental_roberta import MentalRobertaExtractor


class SemanticExtractor(FeatureExtractorBase):
    DIM = 768

    def __init__(self, model=None, tokenizer=None, model_dir: Path | str | None = None):
        self.mental_roberta = MentalRobertaExtractor(
            model=model, tokenizer=tokenizer, model_dir=model_dir
        )

    def extract(self, text: str) -> np.ndarray:
        return self._validate(self.mental_roberta.extract(text))

    @property
    def feature_names(self) -> List[str]:
        return self.mental_roberta.feature_names

    @property
    def sub_extractors(self) -> Dict[str, FeatureExtractorBase]:
        return {"mental_roberta": self.mental_roberta}


__all__ = ["SemanticExtractor"]
