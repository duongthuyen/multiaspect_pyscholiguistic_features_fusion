from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from scripts.config import (
    AFFECTIVE_FEATURES_DIR,
    FEATURES_DIR,
    LEXICAL_FEATURES_DIR,
    SEMANTIC_FEATURES_DIR,
    STRUCTURAL_FEATURES_DIR,
    SYNTACTIC_FEATURES_DIR,
)
from scripts.features.affective import AffectiveExtractor
from scripts.features.base import FeatureExtractorBase
from scripts.features.lexical import LexicalExtractor
from scripts.features.semantic import SemanticExtractor
from scripts.features.structural import StructuralExtractor
from scripts.features.syntactic import SyntacticExtractor

logger = logging.getLogger(__name__)


class FeatureOrchestrator:
    def __init__(self, groups: Dict[str, FeatureExtractorBase] | None = None):
        self.groups = groups or {
            "semantic": SemanticExtractor(),
            "lexical": LexicalExtractor(),
            "syntactic": SyntacticExtractor(),
            "structural": StructuralExtractor(),
            "affective": AffectiveExtractor(),
        }
        self.group_dirs = {
            "semantic": SEMANTIC_FEATURES_DIR,
            "lexical": LEXICAL_FEATURES_DIR,
            "syntactic": SYNTACTIC_FEATURES_DIR,
            "structural": STRUCTURAL_FEATURES_DIR,
            "affective": AFFECTIVE_FEATURES_DIR,
        }

    def extract_all(self, text: str) -> Dict[str, np.ndarray]:
        return {name: extractor.extract(text) for name, extractor in self.groups.items()}

    def parse_components(self, components: str | None) -> Dict[str, set[str] | None]:
        if not components:
            return {group: None for group in self.groups}

        selected: Dict[str, set[str] | None] = {}
        for item in [part.strip() for part in components.split(",") if part.strip()]:
            if "." in item:
                group_name, sub_name = item.split(".", 1)
                if group_name not in self.groups:
                    raise ValueError(f"Unknown feature group: {group_name}")
                if sub_name not in self.groups[group_name].sub_extractors:
                    raise ValueError(f"Unknown sub-extractor: {item}")
                selected.setdefault(group_name, set())
                if selected[group_name] is not None:
                    selected[group_name].add(sub_name)
            else:
                if item not in self.groups:
                    raise ValueError(f"Unknown feature group: {item}")
                selected[item] = None
        return selected

    def extract_dataset(
        self,
        df: pd.DataFrame,
        text_col: str,
        post_id_col: str,
        components: str | None = None,
        force: bool = False,
        split: str | None = None,
    ) -> Dict[str, Dict[str, Path]]:
        if text_col not in df.columns:
            raise KeyError(f"Missing text column: {text_col}")
        if post_id_col not in df.columns:
            raise KeyError(f"Missing post_id column: {post_id_col}")

        selected = self.parse_components(components)
        outputs: Dict[str, Dict[str, Path]] = {}
        meta = {
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(df)),
            "model_versions": self._model_versions(),
            "dimensions": {},
        }

        for group_name, sub_names in selected.items():
            group = self.groups[group_name]
            group_dir = self.group_dirs[group_name]
            if split:
                group_dir = group_dir / split
            group_dir.mkdir(parents=True, exist_ok=True)
            outputs[group_name] = {}

            for sub_name, sub_extractor in group.sub_extractors.items():
                if sub_names is not None and sub_name not in sub_names:
                    continue
                output_path = group_dir / f"{sub_name}.parquet"
                meta["dimensions"][f"{group_name}.{sub_name}"] = sub_extractor.DIM
                if output_path.exists() and not force:
                    logger.info("Skipping %s.%s (already exists)", group_name, sub_name)
                    outputs[group_name][sub_name] = output_path
                    continue
                logger.info("Extracting %s.%s for %d rows...", group_name, sub_name, len(df))
                post_ids = df[post_id_col].tolist()
                texts = df[text_col].astype(str).tolist()
                if hasattr(sub_extractor, "extract_batch"):
                    matrix = sub_extractor.extract_batch(texts)
                    rows = [
                        {"post_id": post_id, "features": features.astype(float).tolist()}
                        for post_id, features in zip(post_ids, matrix)
                    ]
                else:
                    rows = [
                        {
                            "post_id": post_id,
                            "features": sub_extractor.extract(text).astype(float).tolist(),
                        }
                        for post_id, text in zip(post_ids, texts)
                    ]
                pd.DataFrame(rows).to_parquet(output_path, index=False)
                outputs[group_name][sub_name] = output_path
                logger.info("Saved %s.%s -> %s", group_name, sub_name, output_path)

        if split:
            split_dir = FEATURES_DIR / split
            split_dir.mkdir(parents=True, exist_ok=True)
            meta["split"] = split
            (split_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            (FEATURES_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return outputs

    def _model_versions(self) -> Dict[str, str]:
        versions = {}
        try:
            import transformers
            versions["transformers"] = transformers.__version__
        except ImportError:
            pass
        try:
            import sentence_transformers
            versions["sentence_transformers"] = sentence_transformers.__version__
        except ImportError:
            pass
        return versions
