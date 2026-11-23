"""Train linear and multimodal house-price models from the cleaned listings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18


NUMERIC_COLUMNS = ["bedrooms", "bathrooms", "floor_area_m2", "land_area_m2"]
CATEGORICAL_COLUMNS = ["city", "suburb", "property_type"]
TEXT_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{1,}")
PRICE_RE = re.compile(
    r"(?:\b(?:USD|US\$|ZAR|ZWL)\s*\$?\s*\d[\d,]*(?:\.\d+)?"
    r"|\$\s*\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
STOP_WORDS = {
    "about", "after", "also", "and", "are", "bed", "bedroom", "bedrooms",
    "been", "being", "but", "for", "from", "has", "have", "her", "his",
    "into", "its", "near", "our", "the", "their", "there", "this", "that",
    "these", "they", "was", "with", "you", "your", "price", "sale",
}
SEED = 42


def _clean_text(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    text = PRICE_RE.sub(" ", text.lower())
    return [
        token for token in TEXT_TOKEN_RE.findall(text)
        if token not in STOP_WORDS
    ]


def _json_list(value: object, field: str, listing_id: object) -> list[str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid {field} JSON for listing_id={listing_id}: {error}"
        ) from error
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON list for listing_id={listing_id}.")
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def _stratified_split(
    indices: np.ndarray,
    bins: np.ndarray,
    held_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    held: list[int] = []
    for label in np.unique(bins[indices]):
        group = indices[bins[indices] == label].copy()
        rng.shuffle(group)
        held_count = min(max(1, int(round(len(group) * held_fraction)),), len(group) - 1)
        held.extend(group[:held_count].tolist())
        kept.extend(group[held_count:].tolist())
    return np.asarray(kept, dtype=np.int64), np.asarray(held, dtype=np.int64)


def _build_structured_features(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    amenities: list[list[str]],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    train_frame = frame.iloc[train_indices]
    numeric_train = train_frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    medians = numeric_train.median().fillna(0.0)
    numeric_filled = numeric_train.fillna(medians)
    means = numeric_filled.mean()
    scales = numeric_filled.std().replace(0, 1).fillna(1.0)

    numeric_all = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric_all = numeric_all.fillna(medians)
    numeric_values = ((numeric_all - means) / scales).to_numpy(dtype=np.float32)
    feature_parts = [numeric_values]
    feature_names = list(NUMERIC_COLUMNS)

    category_state: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        train_values = train_frame[column].fillna("Unknown").astype(str)
        if column == "suburb":
            top_values = train_values.value_counts().head(50).index.tolist()
            levels = sorted(set(top_values) | {"Other", "Unknown"})
        else:
            levels = sorted(set(train_values.tolist()) | {"Other", "Unknown"})
        category_state[column] = levels

        values = frame[column].fillna("Unknown").astype(str).to_numpy()
        if column == "suburb":
            top_set = set(levels) - {"Other", "Unknown"}
            values = np.asarray(
                [value if value in top_set or value == "Unknown" else "Other"
                 for value in values],
                dtype=object,
            )
        values = np.where(np.isin(values, levels), values, "Other")
        one_hot = (values[:, None] == np.asarray(levels, dtype=object)).astype(np.float32)
        feature_parts.append(one_hot)
        feature_names.extend(f"{column}={level}" for level in levels)

    amenity_counts = Counter(
        amenity
        for index in train_indices
        for amenity in amenities[index]
    )
    amenity_levels = [
        name for name, _ in sorted(
            amenity_counts.items(), key=lambda item: (-item[1], item[0])
        )[:40]
    ]
    amenity_set = set(amenity_levels)
    amenity_features = np.asarray(
        [
            [float(name in amenity_set and name in row) for name in amenity_levels]
            for row in amenities
        ],
        dtype=np.float32,
    )
    feature_parts.append(amenity_features)
    feature_names.extend(f"amenity={name}" for name in amenity_levels)

    state = {
        "numeric_medians": medians.to_dict(),
        "numeric_means": means.to_dict(),
        "numeric_scales": scales.to_dict(),
        "category_levels": category_state,
        "amenity_levels": amenity_levels,
    }
    return np.concatenate(feature_parts, axis=1), feature_names, state


def _make_tfidf(
    tokens: list[list[str]],
    train_indices: np.ndarray,
    max_features: int = 800,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    document_frequency: Counter[str] = Counter()
    for index in train_indices:
        document_frequency.update(set(tokens[index]))
    vocabulary = [
        word for word, count in sorted(
            document_frequency.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 2
    ][:max_features]
    if not vocabulary:
        raise ValueError("No text vocabulary could be built from the training split.")

    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    train_count = len(train_indices)
    idf = np.asarray(
        [
            math.log((1 + train_count) / (1 + document_frequency[word])) + 1
            for word in vocabulary
        ],
        dtype=np.float32,
    )
    matrix = np.zeros((len(tokens), len(vocabulary)), dtype=np.float32)
    for row_index, words in enumerate(tokens):
        counts = Counter(words)
        token_count = max(len(words), 1)
        for word, count in counts.items():
            column_index = word_to_index.get(word)
            if column_index is not None:
                matrix[row_index, column_index] = (
                    count / token_count * idf[column_index]
                )
        norm = np.linalg.norm(matrix[row_index])
        if norm > 0:
            matrix[row_index] /= norm
    return matrix, vocabulary, idf


def _build_text_sequences(
    tokens: list[list[str]],
    train_indices: np.ndarray,
    max_vocabulary: int = 12000,
    max_tokens: int = 192,
) -> tuple[np.ndarray, list[str]]:
    word_counts: Counter[str] = Counter(
        word for index in train_indices for word in tokens[index]
    )
    vocabulary = [
        word for word, count in sorted(
            word_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 2
    ][:max_vocabulary]
    word_to_id = {word: index + 2 for index, word in enumerate(vocabulary)}
    sequences = np.zeros((len(tokens), max_tokens), dtype=np.int64)
    for row_index, words in enumerate(tokens):
        ids = [word_to_id.get(word, 1) for word in words[:max_tokens]]
        if ids:
            sequences[row_index, :len(ids)] = ids
    return sequences, vocabulary


def _image_cache_key(
    listing_ids: list[str], image_paths: list[Path | None]
) -> str:
    parts = ["resnet18-imagenet1k-v1"]
    for listing_id, path in zip(listing_ids, image_paths):
        if path is None:
            parts.append(f"{listing_id}:missing")
        else:
            stat = path.stat()
            parts.append(f"{listing_id}:{path}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _extract_image_features(
    image_paths: list[Path | None],
    listing_ids: list[str],
    cache_dir: Path,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _image_cache_key(listing_ids, image_paths)
    cache_path = cache_dir / f"resnet18_{cache_key}.npz"
    image_present = np.asarray([path is not None for path in image_paths], dtype=np.float32)

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_ids = cached["listing_ids"].astype(str).tolist()
            if cached_ids == listing_ids:
                return cached["features"].astype(np.float32), image_present, str(cache_path)

    if not image_present.any():
        raise ValueError("No local property images were found for the USD listings.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    cnn = resnet18(weights=weights)
    cnn.fc = nn.Identity()
    cnn.to(device).eval()

    features = np.zeros((len(image_paths), 512), dtype=np.float32)
    pending_indices: list[int] = []
    pending_images: list[torch.Tensor] = []

    def encode_pending() -> None:
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(device)
        with torch.no_grad():
            encoded = cnn(batch).cpu().numpy().astype(np.float32)
        features[pending_indices] = encoded
        pending_indices.clear()
        pending_images.clear()

    for index, path in enumerate(image_paths):
        if path is None:
            continue
        with Image.open(path) as image:
            pending_images.append(transform(image.convert("RGB")))
        pending_indices.append(index)
        if len(pending_images) >= batch_size:
            encode_pending()
    encode_pending()

    np.savez_compressed(
        cache_path,
        listing_ids=np.asarray(listing_ids, dtype=str),
        features=features,
    )
    return features, image_present, str(cache_path)


class MultimodalRegressor(nn.Module):
    def __init__(
        self,
        structured_dim: int,
        vocabulary_size: int,
        baseline_coefficients: np.ndarray,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(
            vocabulary_size + 2, embedding_dim, padding_idx=0
        )
        self.register_buffer(
            "linear_coefficients",
            torch.as_tensor(baseline_coefficients[1:], dtype=torch.float32),
        )
        self.register_buffer(
            "linear_intercept",
            torch.as_tensor(float(baseline_coefficients[0]), dtype=torch.float32),
        )
        self.structured_branch = nn.Sequential(
            nn.Linear(structured_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.text_branch = nn.Sequential(
            nn.Linear(embedding_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.image_branch = nn.Sequential(
            nn.Linear(513, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.residual = nn.Sequential(
            nn.Linear(96 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        baseline_features: torch.Tensor,
        structured: torch.Tensor,
        image_features: torch.Tensor,
        image_present: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = tokens.ne(0).unsqueeze(-1)
        token_vectors = self.text_embedding(tokens)
        pooled_text = (token_vectors * token_mask).sum(dim=1)
        pooled_text = pooled_text / token_mask.sum(dim=1).clamp(min=1)
        structured_vector = self.structured_branch(structured)
        text_vector = self.text_branch(pooled_text)
        image_vector = self.image_branch(
            torch.cat([image_features, image_present.unsqueeze(1)], dim=1)
        )
        linear_prediction = (
            baseline_features @ self.linear_coefficients + self.linear_intercept
        )
        residual_prediction = self.residual(
            torch.cat([structured_vector, text_vector, image_vector], dim=1)
        ).squeeze(1)
        return linear_prediction + residual_prediction


def _metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    actual = np.expm1(actual_log).clip(min=0)
    predicted = np.expm1(predicted_log).clip(min=0)
    error = predicted - actual
    total_variance = np.square(actual - actual.mean()).sum()
    r_squared = (
        1.0 - float(np.square(error).sum()) / float(total_variance)
        if total_variance > 0 else float("nan")
    )
    return {
        "MAE_USD": float(np.abs(error).mean()),
        "RMSE_USD": float(np.sqrt(np.square(error).mean())),
        "R2_USD": r_squared,
        "MAE_log": float(np.abs(predicted_log - actual_log).mean()),
    }


def _build_residual_diagnostics(
    prediction_frame: pd.DataFrame,
    report_dir: Path,
) -> dict[str, object]:
    model_columns = {
        "Linear Ridge": "linear_baseline_price_usd",
        "Multimodal": "multimodal_price_usd",
    }
    actual = prediction_frame["actual_price_usd"].to_numpy(dtype=np.float64)
    actual_log = np.log1p(actual)
    residual_frames = []
    summary_rows = []
    band_rows = []

    for model_name, prediction_column in model_columns.items():
        predicted = prediction_frame[prediction_column].to_numpy(dtype=np.float64)
        residual = actual - predicted
        log_residual = actual_log - np.log1p(predicted)
        absolute_error = np.abs(residual)
        residual_frames.append(pd.DataFrame({
            "listing_id": prediction_frame["listing_id"].to_numpy(),
            "model": model_name,
            "actual_price_usd": actual,
            "predicted_price_usd": predicted,
            "residual_usd": residual,
            "absolute_error_usd": absolute_error,
            "residual_log": log_residual,
        }))
        summary_rows.append({
            "model": model_name,
            "n": len(actual),
            "mean_residual_usd": float(residual.mean()),
            "median_residual_usd": float(np.median(residual)),
            "p10_residual_usd": float(np.quantile(residual, 0.10)),
            "p90_residual_usd": float(np.quantile(residual, 0.90)),
            "p90_absolute_error_usd": float(np.quantile(absolute_error, 0.90)),
            "mean_residual_log": float(log_residual.mean()),
            "median_residual_log": float(np.median(log_residual)),
        })

        price_bands = pd.qcut(actual, q=5, duplicates="drop")
        band_frame = pd.DataFrame({
            "price_band": price_bands.astype(str),
            "actual_price_usd": actual,
            "residual_usd": residual,
        })
        for band, group in band_frame.groupby("price_band", sort=False):
            errors = group["residual_usd"].to_numpy()
            band_rows.append({
                "model": model_name,
                "actual_price_band": band,
                "n": len(group),
                "actual_min_usd": float(group["actual_price_usd"].min()),
                "actual_max_usd": float(group["actual_price_usd"].max()),
                "MAE_USD": float(np.abs(errors).mean()),
                "RMSE_USD": float(np.sqrt(np.square(errors).mean())),
                "mean_residual_usd": float(errors.mean()),
            })

    residuals = pd.concat(residual_frames, ignore_index=True)
    residual_metrics = pd.DataFrame(summary_rows)
    errors_by_price_band = pd.DataFrame(band_rows)
    residuals_path = report_dir / "test_residuals.csv"
    residual_metrics_path = report_dir / "residual_metrics.csv"
    price_band_path = report_dir / "errors_by_price_band.csv"
    residual_plot_path = report_dir / "residual_diagnostics.png"
    residuals.to_csv(residuals_path, index=False)
    residual_metrics.to_csv(residual_metrics_path, index=False)
    errors_by_price_band.to_csv(price_band_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, model_name in enumerate(model_columns):
        model_residuals = residuals.loc[residuals["model"].eq(model_name)]
        axes[row, 0].scatter(
            model_residuals["predicted_price_usd"],
            model_residuals["residual_usd"],
            alpha=0.4,
            s=18,
        )
        axes[row, 0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[row, 0].set_xscale("log")
        axes[row, 0].set_yscale("symlog", linthresh=10_000)
        axes[row, 0].set(
            title=f"{model_name}: USD residual vs fitted",
            xlabel="Predicted price (USD, log scale)",
            ylabel="Residual: actual - predicted (USD, symlog scale)",
        )
        axes[row, 1].hist(model_residuals["residual_log"], bins=30, alpha=0.8)
        axes[row, 1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[row, 1].set(
            title=f"{model_name}: log residual distribution",
            xlabel="Residual: log1p(actual) - log1p(predicted)",
            ylabel="Listings",
        )
    fig.tight_layout()
    fig.savefig(residual_plot_path, dpi=160)
    plt.close(fig)

    return {
        "residuals": residuals,
        "residual_metrics": residual_metrics,
        "errors_by_price_band": errors_by_price_band,
        "paths": {
            "residuals": residuals_path,
            "residual_metrics": residual_metrics_path,
            "errors_by_price_band": price_band_path,
            "residual_plot": residual_plot_path,
        },
    }


def _train_fusion_model(
    baseline_features: np.ndarray,
    structured: np.ndarray,
    image_features: np.ndarray,
    image_present: np.ndarray,
    sequences: np.ndarray,
    target_log: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    vocabulary_size: int,
    baseline_coefficients: np.ndarray,
    max_epochs: int,
    seed: int,
) -> tuple[MultimodalRegressor, list[dict[str, float]], int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalRegressor(
        structured.shape[1],
        vocabulary_size,
        baseline_coefficients,
    ).to(device)
    loss_function = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tensors(indices: np.ndarray) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(baseline_features[indices]),
            torch.from_numpy(structured[indices]),
            torch.from_numpy(image_features[indices]),
            torch.from_numpy(image_present[indices]),
            torch.from_numpy(sequences[indices]),
            torch.from_numpy(target_log[indices]),
        )

    train_data = TensorDataset(*tensors(train_indices))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_data, batch_size=64, shuffle=True, generator=generator, num_workers=0
    )
    validation_tensors = tuple(tensor.to(device) for tensor in tensors(validation_indices))
    history: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        initial_prediction = model(*validation_tensors[:-1])
        best_loss = float(
            loss_function(initial_prediction, validation_tensors[-1]).item()
        )
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    best_epoch = 0
    patience = 10
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        observations = 0
        for baseline_batch, structured_batch, image_batch, present_batch, token_batch, target_batch in loader:
            baseline_batch = baseline_batch.to(device)
            structured_batch = structured_batch.to(device)
            image_batch = image_batch.to(device)
            present_batch = present_batch.to(device)
            token_batch = token_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                baseline_batch,
                structured_batch,
                image_batch,
                present_batch,
                token_batch,
            )
            loss = loss_function(prediction, target_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(target_batch)
            observations += len(target_batch)

        model.eval()
        with torch.no_grad():
            val_prediction = model(*validation_tensors[:-1])
            validation_loss = float(
                loss_function(val_prediction, validation_tensors[-1]).item()
            )
        training_loss = total_loss / max(observations, 1)
        history.append({
            "epoch": epoch,
            "train_loss": training_loss,
            "validation_loss": validation_loss,
        })

        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, history, best_epoch


def run_training(
    project_root: Path,
    max_epochs: int = 60,
    seed: int = SEED,
) -> dict[str, object]:
    """Fit the linear and multimodal models and save metrics, predictions, and weights."""
    project_root = Path(project_root).resolve()
    data_path = project_root / "reports" / "eda" / "cleaned_listings.csv"
    report_dir = project_root / "reports" / "modeling"
    model_dir = project_root / "models"
    cache_dir = model_dir / "cache"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}; run notebooks/01_clean_explore.ipynb first."
        )
    frame = pd.read_csv(data_path)
    frame["price_usd"] = pd.to_numeric(frame["price_usd"], errors="coerce")
    frame = frame.loc[frame["price_usd"].gt(0)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No positive USD prices are available for model training.")
    if frame["property_url"].duplicated().any():
        raise ValueError("Duplicate property_url rows remain in the cleaned dataset.")

    listing_ids = frame["listing_id"].astype(str).tolist()
    target = frame["price_usd"].to_numpy(dtype=np.float64)
    target_log = np.log1p(target).astype(np.float32)
    price_bins = np.asarray(
        pd.qcut(target_log, q=10, labels=False, duplicates="drop"),
        dtype=np.int64,
    )
    all_indices = np.arange(len(frame), dtype=np.int64)
    train_validation_indices, test_indices = _stratified_split(
        all_indices, price_bins, held_fraction=0.20, seed=seed
    )
    train_indices, validation_indices = _stratified_split(
        train_validation_indices,
        price_bins,
        held_fraction=0.15,
        seed=seed + 1,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    amenities = [
        _json_list(value, "amenities_json", listing_id)
        for value, listing_id in zip(frame["amenities_json"], listing_ids)
    ]
    structured, structured_names, structured_state = _build_structured_features(
        frame, train_indices, amenities
    )
    text_tokens = [
        _clean_text(f"{title} {description}")
        for title, description in zip(frame["title"], frame["description"])
    ]

    tfidf, tfidf_vocabulary, tfidf_idf = _make_tfidf(
        text_tokens, train_indices, max_features=800
    )
    baseline_features = np.concatenate([structured, tfidf], axis=1)
    baseline_names = structured_names + [f"text_tfidf={word}" for word in tfidf_vocabulary]
    baseline_train = np.column_stack([
        np.ones(len(train_indices), dtype=np.float32),
        baseline_features[train_indices],
    ])
    baseline_test = np.column_stack([
        np.ones(len(test_indices), dtype=np.float32),
        baseline_features[test_indices],
    ])
    alpha = 1.0
    gram = baseline_train.T @ baseline_train
    gram.flat[::gram.shape[0] + 1] += alpha
    gram[0, 0] -= alpha
    baseline_coefficients = np.linalg.solve(
        gram, baseline_train.T @ target_log[train_indices]
    )
    baseline_test_log = baseline_test @ baseline_coefficients
    baseline_metrics = _metrics(target_log[test_indices], baseline_test_log)

    image_paths: list[Path | None] = []
    for row, listing_id in zip(frame.itertuples(index=False), listing_ids):
        photo_paths = _json_list(row.photo_paths_json, "photo_paths_json", listing_id)
        resolved = [
            (project_root / photo_path).resolve()
            for photo_path in photo_paths
        ]
        existing = next((path for path in resolved if path.is_file()), None)
        image_paths.append(existing)

    image_features, image_present, cache_path = _extract_image_features(
        image_paths, listing_ids, cache_dir
    )
    image_mean = image_features[train_indices].mean(axis=0)
    image_scale = image_features[train_indices].std(axis=0)
    image_scale[image_scale < 1e-6] = 1.0
    image_features = ((image_features - image_mean) / image_scale).astype(np.float32)

    sequences, text_vocabulary = _build_text_sequences(text_tokens, train_indices)
    multimodal, history, best_epoch = _train_fusion_model(
        baseline_features,
        structured,
        image_features,
        image_present,
        sequences,
        target_log,
        train_indices,
        validation_indices,
        vocabulary_size=len(text_vocabulary),
        baseline_coefficients=baseline_coefficients,
        max_epochs=max_epochs,
        seed=seed,
    )
    device = next(multimodal.parameters()).device
    test_tensors = (
        torch.from_numpy(baseline_features[test_indices]).to(device),
        torch.from_numpy(structured[test_indices]).to(device),
        torch.from_numpy(image_features[test_indices]).to(device),
        torch.from_numpy(image_present[test_indices]).to(device),
        torch.from_numpy(sequences[test_indices]).to(device),
    )
    with torch.no_grad():
        multimodal_test_log = multimodal(*test_tensors).cpu().numpy()
    multimodal_metrics = _metrics(target_log[test_indices], multimodal_test_log)

    metrics = pd.DataFrame([
        {"model": "Linear Ridge (structured + TF-IDF)", **baseline_metrics},
        {"model": "Multimodal ResNet18 + text embeddings + MLP", **multimodal_metrics},
    ]).sort_values("MAE_USD").reset_index(drop=True)
    metrics_path = report_dir / "model_comparison.csv"
    metrics.to_csv(metrics_path, index=False)

    prediction_frame = pd.DataFrame({
        "listing_id": np.asarray(listing_ids)[test_indices],
        "actual_price_usd": target[test_indices],
        "linear_baseline_price_usd": np.expm1(baseline_test_log).clip(min=0),
        "multimodal_price_usd": np.expm1(multimodal_test_log).clip(min=0),
    })
    prediction_path = report_dir / "test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    residual_diagnostics = _build_residual_diagnostics(prediction_frame, report_dir)

    history_frame = pd.DataFrame(history)
    history_path = report_dir / "training_history.csv"
    history_frame.to_csv(history_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual = prediction_frame["actual_price_usd"].to_numpy()
    for axis, column, title in [
        (axes[0], "linear_baseline_price_usd", "Linear Ridge baseline"),
        (axes[1], "multimodal_price_usd", "Multimodal fusion model"),
    ]:
        predicted = prediction_frame[column].to_numpy()
        axis.scatter(actual, predicted, alpha=0.45, s=22)
        limits = [max(1.0, float(min(actual.min(), predicted.min()))),
                  float(max(actual.max(), predicted.max()))]
        axis.plot(limits, limits, "k--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(title=title, xlabel="Actual price (USD)", ylabel="Predicted price (USD)")
    fig.suptitle("Held-out test predictions")
    fig.tight_layout()
    prediction_plot_path = report_dir / "actual_vs_predicted.png"
    fig.savefig(prediction_plot_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
    ax.plot(history_frame["epoch"], history_frame["validation_loss"], label="Validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}")
    ax.set(title="Multimodal training loss (log-price)", xlabel="Epoch", ylabel="Smooth L1 loss")
    ax.legend()
    fig.tight_layout()
    history_plot_path = report_dir / "training_history.png"
    fig.savefig(history_plot_path, dpi=160)
    plt.close(fig)

    linear_path = model_dir / "linear_baseline.npz"
    np.savez_compressed(
        linear_path,
        coefficients=baseline_coefficients,
        feature_names=np.asarray(["intercept", *baseline_names], dtype=str),
        tfidf_vocabulary=np.asarray(tfidf_vocabulary, dtype=str),
        tfidf_idf=tfidf_idf,
    )
    multimodal_path = model_dir / "multimodal_fusion.pt"
    torch.save({
        "state_dict": multimodal.state_dict(),
        "structured_dim": int(structured.shape[1]),
        "text_vocabulary": text_vocabulary,
        "structured_state": structured_state,
        "image_mean": image_mean,
        "image_scale": image_scale,
        "cnn_weights": "ResNet18_Weights.DEFAULT (frozen feature extractor)",
        "target_transform": "log1p(price_usd)",
        "seed": seed,
        "best_epoch": best_epoch,
    }, multimodal_path)

    metadata = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(len(test_indices)),
        "image_rows": int(image_present.sum()),
        "image_cache": str(Path(cache_path).relative_to(project_root)),
        "seed": seed,
        "target_transform": "log1p(price_usd)",
        "baseline_features": "numeric, city/suburb/property type, top amenities, train-only TF-IDF",
        "multimodal_architecture": (
            "fixed linear baseline skip + frozen ImageNet ResNet18 image features "
            "+ learned masked-mean word embeddings + structured MLP branches "
            "+ trainable multimodal residual fusion MLP"
        ),
        "best_epoch": best_epoch,
        "baseline_model": str(linear_path.relative_to(project_root)),
        "multimodal_model": str(multimodal_path.relative_to(project_root)),
        "metrics": metrics.to_dict(orient="records"),
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Rows: {len(frame):,}; train/validation/test: "
          f"{len(train_indices):,}/{len(validation_indices):,}/{len(test_indices):,}")
    print(f"Listings with a local image: {int(image_present.sum()):,}/{len(frame):,}")
    print(f"Multimodal best epoch: {best_epoch}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"Reports: {report_dir}")
    print(f"Saved models: {linear_path}, {multimodal_path}")

    return {
        "metrics": metrics,
        "history": history_frame,
        "predictions": prediction_frame,
        "residuals": residual_diagnostics["residuals"],
        "residual_metrics": residual_diagnostics["residual_metrics"],
        "errors_by_price_band": residual_diagnostics["errors_by_price_band"],
        "metadata": metadata,
        "paths": {
            "metrics": metrics_path,
            "predictions": prediction_path,
            **residual_diagnostics["paths"],
            "training_history": history_path,
            "prediction_plot": prediction_plot_path,
            "history_plot": history_plot_path,
            "linear_model": linear_path,
            "multimodal_model": multimodal_path,
            "metadata": metadata_path,
            "image_cache": Path(cache_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_training(args.project_root, max_epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
