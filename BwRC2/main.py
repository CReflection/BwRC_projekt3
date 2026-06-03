import time
import os

from pynput import keyboard
import matplotlib.pyplot as plt

from keystroke_system import (
    build_model,
    build_vocabulary_from_sentences,
    extract_features,
    identify,
    leave_one_out,
    load_raw_dataset,
    METRICS,
    verify,
    estimate_threshold,
)

BASE_DIR = "data"
SENTENCES_FILE = "sentences.txt"

DEFAULT_K = 3
DEFAULT_METRIC = "braycurtis"


########################################
# Utils
########################################

def flush_input():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass


def load_sentences():
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_vocab():
    sentences = load_sentences()
    return build_vocabulary_from_sentences(sentences)


def ensure_data_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def choose_parameters():
    try:
        raw_k = input(f"k [{DEFAULT_K}]: ").strip()
        k = int(raw_k) if raw_k else DEFAULT_K
    except ValueError:
        k = DEFAULT_K

    metric = input(f"metryka {list(METRICS.keys())} [{DEFAULT_METRIC}]: ").strip()
    if not metric:
        metric = DEFAULT_METRIC
    if metric not in METRICS:
        metric = DEFAULT_METRIC

    threshold_raw = input("próg [auto]: ").strip()
    threshold = None
    if threshold_raw:
        try:
            threshold = float(threshold_raw)
        except ValueError:
            threshold = None

    return k, metric, threshold


########################################
# Capture
########################################

def capture_events():
    sentences = load_sentences()
    events = []

    sentence_index = 0
    typed_text = ""

    def show_next():
        nonlocal typed_text
        typed_text = ""
        print(f"\n[Zadanie {sentence_index + 1}/{len(sentences)}]")
        print(sentences[sentence_index])
        print("> ", end="", flush=True)

    def on_press(key):
        nonlocal typed_text

        try:
            if hasattr(key, "char") and key.char:
                key_name = key.char
            else:
                key_name = str(key)
        except Exception:
            key_name = str(key)

        if key == keyboard.Key.space:
            key_name = " "
        elif key == keyboard.Key.backspace:
            key_name = "<BACKSPACE>"
        elif key == keyboard.Key.enter:
            key_name = "<ENTER>"
        elif key == keyboard.Key.esc:
            key_name = "<ESC>"
        elif key_name.startswith("Key."):
            key_name = f"<{key_name[4:].upper()}>"

        events.append((time.time(), key_name, "keydown"))

        if hasattr(key, "char") and key.char:
            typed_text += key.char
            print(key.char, end="", flush=True)
        elif key == keyboard.Key.space:
            typed_text += " "
            print(" ", end="", flush=True)
        elif key == keyboard.Key.backspace:
            if typed_text:
                typed_text = typed_text[:-1]
                print("\b \b", end="", flush=True)
        elif key == keyboard.Key.enter:
            print()

    def on_release(key):
        nonlocal sentence_index

        try:
            if hasattr(key, "char") and key.char:
                key_name = key.char
            else:
                key_name = str(key)
        except Exception:
            key_name = str(key)

        if key == keyboard.Key.space:
            key_name = " "
        elif key == keyboard.Key.backspace:
            key_name = "<BACKSPACE>"
        elif key == keyboard.Key.enter:
            key_name = "<ENTER>"
        elif key == keyboard.Key.esc:
            key_name = "<ESC>"
        elif key_name.startswith("Key."):
            key_name = f"<{key_name[4:].upper()}>"

        events.append((time.time(), key_name, "keyup"))

        if key == keyboard.Key.enter:
            if typed_text.strip() == "":
                return

            sentence_index += 1
            if sentence_index >= len(sentences):
                return False

            show_next()

        if key == keyboard.Key.esc:
            return False

    show_next()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    flush_input()
    return events


def capture_sample(vocab):
    return extract_features(capture_events(), vocab)


########################################
# Registration
########################################

def collect_data(username):
    ensure_data_dir()

    user_dir = os.path.join(BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    base_n = 1
    while os.path.exists(os.path.join(user_dir, f"{username}_{base_n}.csv")):
        base_n += 1

    print(f"\n🎯 Rejestracja: {username}")
    print("Każdy użytkownik powinien mieć co najmniej 5 próbek z różnych tekstów.")

    for i in range(5):
        print(f"\n--- Próba {i + 1}/5 ---")
        events = capture_events()

        file_path = os.path.join(user_dir, f"{username}_{base_n + i}.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            for t, key, event in events:
                f.write(f"{t};{key};{event}\n")

        print(f"✔ zapisano: {file_path}")


########################################
# Identification / Verification
########################################

def identify_user():
    print("\n✍️ Identyfikacja:")
    vocab = get_vocab()
    sample = capture_sample(vocab)

    k, metric, threshold_input = choose_parameters()
    model = build_model(BASE_DIR, vocab)

    if len(model["X"]) == 0:
        print("\n❌ Brak danych w bazie.")
        return

    threshold = threshold_input
    if threshold is None:
        threshold = estimate_threshold(model, k=k, metric=metric, percentile=97.5, margin=1.15)

    print(f"\nℹ️ Użyty próg: {threshold:.4f}")

    user, nearest_dist, avg_dist = identify(
        model,
        sample,
        k=k,
        metric=metric,
        threshold=threshold,
    )

    if user == "UNKNOWN":
        print(f"\n❌ Nie rozpoznano (nearest_dist={nearest_dist:.4f}, avg_dist={avg_dist:.4f})")
    else:
        print(f"\n✅ Użytkownik: {user} (nearest_dist={nearest_dist:.4f}, avg_dist={avg_dist:.4f})")


def verify_user():
    username = input("Podaj użytkownika: ").strip()
    if not username:
        print("\n❌ Nie podano użytkownika.")
        return

    print("\n✍️ Logowanie:")
    vocab = get_vocab()
    sample = capture_sample(vocab)

    k, metric, threshold_input = choose_parameters()
    model = build_model(BASE_DIR, vocab)

    if len(model["X"]) == 0:
        print("\n❌ Brak danych w bazie.")
        return

    threshold = threshold_input
    if threshold is None:
        threshold = estimate_threshold(model, k=k, metric=metric, percentile=97.5, margin=1.15)

    print(f"\nℹ️ Użyty próg: {threshold:.4f}")

    ok, nearest_dist, avg_dist = verify(
        model,
        sample,
        username,
        k=k,
        metric=metric,
        threshold=threshold,
    )

    if ok:
        print(f"\n✅ Dostęp OK (nearest_dist={nearest_dist:.4f}, avg_dist={avg_dist:.4f})")
    else:
        print(f"\n❌ Dostęp odrzucony (nearest_dist={nearest_dist:.4f}, avg_dist={avg_dist:.4f})")


########################################
# Test + wykres
########################################

def test_and_plot():
    vocab = get_vocab()
    X_raw, y = load_raw_dataset(BASE_DIR, vocab)

    if len(X_raw) == 0:
        print("\n❌ Brak danych do testu.")
        return

    k_values = [1, 3, 5, 7, 9]
    k_values = [k for k in k_values if k < len(X_raw)]
    if not k_values:
        k_values = [1]

    metrics = list(METRICS.keys())
    results = {}

    print("\n🔬 Test leave-one-out")

    for metric in metrics:
        results[metric] = []
        for k in k_values:
            acc = leave_one_out(X_raw, y, k=k, metric=metric)
            results[metric].append(acc)
            print(f"metric={metric}, k={k} -> accuracy={acc:.4f}")

    plt.figure(figsize=(9, 5))
    for metric in metrics:
        plt.plot(k_values, results[metric], marker="o", linewidth=2, label=metric)

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Leave-One-Out: accuracy vs k")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = "accuracy_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.show()

    print(f"\n✔ Wykres zapisano do: {out_path}")


########################################
# Menu
########################################

def menu():
    print("\n=== KEYSTROKE SYSTEM ===")
    print("1. Rejestracja")
    print("2. Identyfikacja")
    print("3. Weryfikacja")
    print("4. Test + wykres")
    print("0. Wyjście")


########################################
# Main
########################################

if __name__ == "__main__":
    ensure_data_dir()

    while True:
        menu()
        flush_input()
        choice = input("Wybór: ").strip()

        if choice == "1":
            username = input("Nazwa użytkownika: ").strip()
            if username:
                collect_data(username)
            else:
                print("❌ Nie podano nazwy użytkownika.")

        elif choice == "2":
            identify_user()

        elif choice == "3":
            verify_user()

        elif choice == "4":
            test_and_plot()

        elif choice == "0":
            break

        else:
            print("❌ Zła opcja")