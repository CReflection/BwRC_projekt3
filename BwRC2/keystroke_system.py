import os
import glob
from collections import Counter

import numpy as np


########################################
# 1. Wczytywanie danych CSV
########################################

def normalize_key_token(key: str) -> str:
    key = key.strip()

    mapping = {
        "Key.space": " ",
        "Key.enter": "<ENTER>",
        "Key.backspace": "<BACKSPACE>",
        "Key.esc": "<ESC>",
        "Key.tab": "<TAB>",
    }

    if key in mapping:
        return mapping[key]

    if key.startswith("Key."):
        return f"<{key[4:].upper()}>"

    return key


def load_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != 3:
                continue

            try:
                t = float(parts[0].strip())
            except ValueError:
                continue

            key = normalize_key_token(parts[1])
            event = parts[2].strip().lower()
            data.append((t, key, event))

    return data


########################################
# 2. Słownik znaków
########################################

def build_vocabulary_from_sentences(sentences):
    chars = set()
    for sentence in sentences:
        for ch in sentence:
            chars.add(ch)
    return sorted(chars)


########################################
# 3. Ekstrakcja cech
########################################

def _is_printable_char_token(key):
    return len(key) == 1 or key == " "


def _mean_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _std_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.std(values))


def extract_pressed_sequence(events):
    """
    Rekonstrukcja sekwencji wpisanych znaków z uwzględnieniem backspace.
    Zwraca listę rekordów:
    {
        "key": znak,
        "down": czas_down,
        "up": czas_up
    }
    """
    records = []

    for t, key, event in sorted(events, key=lambda x: x[0]):
        if key in {"<ENTER>", "<ESC>", "<TAB>"}:
            continue

        if key == "<BACKSPACE>" and event == "keydown":
            for idx in range(len(records) - 1, -1, -1):
                if _is_printable_char_token(records[idx]["key"]):
                    records.pop(idx)
                    break
            continue

        if not _is_printable_char_token(key):
            continue

        if event == "keydown":
            records.append({"key": key, "down": t, "up": None})
        elif event == "keyup":
            for idx in range(len(records) - 1, -1, -1):
                if records[idx]["key"] == key and records[idx]["up"] is None:
                    records[idx]["up"] = t
                    break

    return [r for r in records if r["up"] is not None and r["down"] is not None]


def extract_features(events, vocab):
    """
    Cechy znakowe:
    dla każdego znaku z vocab liczymy:
    - count
    - mean dwell
    - std dwell
    - mean flight
    - std flight

    Wektor ma stałą długość: 5 * len(vocab)
    """
    vocab = list(vocab)
    accepted = extract_pressed_sequence(events)

    dwell_map = {ch: [] for ch in vocab}
    flight_map = {ch: [] for ch in vocab}
    count_map = {ch: 0 for ch in vocab}

    for i, rec in enumerate(accepted):
        key = rec["key"]
        dwell = float(rec["up"] - rec["down"])

        if i + 1 < len(accepted):
            flight = float(accepted[i + 1]["down"] - rec["up"])
        else:
            flight = 0.0

        if key in count_map:
            count_map[key] += 1
            dwell_map[key].append(dwell)
            flight_map[key].append(flight)

    features = []
    for ch in vocab:
        features.extend([
            float(count_map[ch]),
            _mean_or_zero(dwell_map[ch]),
            _std_or_zero(dwell_map[ch]),
            _mean_or_zero(flight_map[ch]),
            _std_or_zero(flight_map[ch]),
        ])

    return np.asarray(features, dtype=float)


########################################
# 4. Metryki
########################################

def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def chebyshev(a, b):
    return float(np.max(np.abs(a - b)))


def bray_curtis(a, b):
    denom = np.sum(np.abs(a + b)) + 1e-10
    return float(np.sum(np.abs(a - b)) / denom)


METRICS = {
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "braycurtis": bray_curtis,
}


########################################
# 5. Skalowanie Min-Max
########################################

def fit_minmax(X):
    X = np.asarray(X, dtype=float)

    if len(X) == 0:
        raise ValueError("Nie można dopasować skalera do pustego zbioru.")

    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    scale = x_max - x_min
    scale[scale == 0.0] = 1.0

    return {
        "min": x_min,
        "scale": scale,
    }


def transform_minmax(X, scaler):
    X = np.asarray(X, dtype=float)
    return (X - scaler["min"]) / scaler["scale"]


########################################
# 6. Zbiór danych
########################################

def load_raw_dataset(base_dir, vocab):
    X = []
    y = []

    if not os.path.isdir(base_dir):
        return np.asarray(X, dtype=float), np.asarray(y, dtype=str)

    for user in sorted(os.listdir(base_dir)):
        user_dir = os.path.join(base_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for file_path in sorted(glob.glob(os.path.join(user_dir, "*.csv"))):
            events = load_csv(file_path)
            if not events:
                continue

            features = extract_features(events, vocab)
            X.append(features)
            y.append(user)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=str)


def build_model(base_dir, vocab):
    """
    Zwraca słownik:
    - X_raw: surowe cechy
    - y: etykiety
    - scaler: parametry min-max
    - X: cechy po skalowaniu
    - vocab: słownik znaków
    """
    X_raw, y = load_raw_dataset(base_dir, vocab)

    model = {
        "X_raw": X_raw,
        "y": y,
        "scaler": None,
        "X": np.asarray([], dtype=float),
        "vocab": list(vocab),
    }

    if len(X_raw) == 0:
        return model

    scaler = fit_minmax(X_raw)
    X = transform_minmax(X_raw, scaler)

    model["scaler"] = scaler
    model["X"] = X
    return model


########################################
# 7. k-NN
########################################

def knn_predict(X_train, y_train, sample, k=3, metric="euclidean"):
    if len(X_train) == 0:
        raise ValueError("Pusty zbiór uczący.")

    if metric not in METRICS:
        metric = "euclidean"

    k = max(1, min(int(k), len(X_train)))
    dist_func = METRICS[metric]

    distances = []
    for xi, label in zip(X_train, y_train):
        d = dist_func(sample, xi)
        distances.append((d, label))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    labels = [label for _, label in neighbors]
    dists = [d for d, _ in neighbors]

    counts = Counter(labels)
    best_count = max(counts.values())
    tied_labels = [lab for lab, cnt in counts.items() if cnt == best_count]

    if len(tied_labels) == 1:
        predicted = tied_labels[0]
    else:
        best_label = None
        best_avg = None
        for lab in tied_labels:
            lab_dists = [d for d, l in neighbors if l == lab]
            avg = float(np.mean(lab_dists))
            if best_avg is None or avg < best_avg:
                best_avg = avg
                best_label = lab
        predicted = best_label

    nearest_dist = float(dists[0])
    avg_dist = float(np.mean(dists))

    return predicted, nearest_dist, avg_dist


########################################
# 8. Leave-One-Out
########################################

def leave_one_out(X_raw, y, k=3, metric="euclidean"):
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=str)

    if len(X_raw) < 2:
        return 0.0

    correct = 0

    for i in range(len(X_raw)):
        X_train_raw = np.delete(X_raw, i, axis=0)
        y_train = np.delete(y, i)

        scaler = fit_minmax(X_train_raw)
        X_train = transform_minmax(X_train_raw, scaler)
        sample = transform_minmax(X_raw[i], scaler)

        pred, _, _ = knn_predict(X_train, y_train, sample, k=k, metric=metric)

        if pred == y[i]:
            correct += 1

    return correct / len(X_raw)


########################################
# 9. Identyfikacja
########################################

def identify(model, sample, k=3, metric="euclidean", threshold=None):
    if model["scaler"] is None or len(model["X"]) == 0:
        return "UNKNOWN", float("inf"), float("inf")

    sample = transform_minmax(sample, model["scaler"])
    pred, nearest_dist, avg_dist = knn_predict(model["X"], model["y"], sample, k=k, metric=metric)

    if threshold is not None and nearest_dist > threshold:
        return "UNKNOWN", nearest_dist, avg_dist

    return pred, nearest_dist, avg_dist


########################################
# 10. Weryfikacja
########################################

def verify(model, sample, claimed_user, k=3, metric="euclidean", threshold=0.5):
    if model["scaler"] is None or len(model["X"]) == 0:
        return False, float("inf"), float("inf")

    sample = transform_minmax(sample, model["scaler"])
    pred, nearest_dist, avg_dist = knn_predict(model["X"], model["y"], sample, k=k, metric=metric)

    ok = (pred == claimed_user) and (nearest_dist <= threshold)
    return ok, nearest_dist, avg_dist


########################################
# 11. Automatyczny próg
########################################

def estimate_threshold(model, k=3, metric="euclidean", percentile=97.5, margin=1.15):
    """
    Szacuje próg na podstawie leave-one-out tylko dla poprawnie rozpoznanych próbek.
    Finalny próg = percentile(poprawne_dystanse) * margin
    """
    X = np.asarray(model["X"], dtype=float)
    y = np.asarray(model["y"], dtype=str)

    if len(X) < 2:
        return 0.5

    correct_distances = []
    all_distances = []

    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        sample = X[i]

        pred, nearest_dist, _ = knn_predict(X_train, y_train, sample, k=k, metric=metric)
        all_distances.append(nearest_dist)

        if pred == y[i]:
            correct_distances.append(nearest_dist)

    if len(correct_distances) == 0:
        base = float(np.percentile(all_distances, percentile))
    else:
        base = float(np.percentile(correct_distances, percentile))

    return float(base * margin)


########################################
# 12. Pomocnicze
########################################

def available_users(base_dir):
    if not os.path.isdir(base_dir):
        return []
    return sorted([u for u in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, u))])