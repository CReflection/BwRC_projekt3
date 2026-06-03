import time
import os
import cv2
import numpy as np

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

# Konfiguracja dla modułu twarzy
FACE_SIZE = (200, 200)  # Eigenfaces WYMAGA identycznych wymiarów zdjęć
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
DEFAULT_FACE_THRESHOLD = 4000.0  # Próg odległości dla Eigenfaces (do dostosowania)


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

    threshold_raw = input("próg klawiatury [auto]: ").strip()
    threshold = None
    if threshold_raw:
        try:
            threshold = float(threshold_raw)
        except ValueError:
            threshold = None

    return k, metric, threshold


########################################
# Pomocnicze funkcje dla Face ID
########################################

def get_user_mappings():
    """Tworzy słownik mapujący nazwy użytkowników na unikalne ID (potrzebne dla OpenCV)."""
    users = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    user_to_id = {user: i for i, user in enumerate(users)}
    id_to_user = {i: user for i, user in enumerate(users)}
    return user_to_id, id_to_user


def capture_face_image():
    """Uruchamia kamerę i próbuje wykryć oraz wyciąć twarz użytkownika."""
    cap = cv2.VideoCapture(0)
    print("[INFO] Spójrz w kamerę i naciśnij [SPACJĘ], aby zrobić zdjęcie twarzy (lub [ESC] aby anulować)...")
    
    face_roi = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Błąd kamery.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Rysowanie ramki wokół wykrytej twarzy
        frame_vis = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow('Zdjecie Twarzy - Spacja by zapisac', frame_vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacja zapisuje
            if len(faces) > 0:
                # Bierzemy pierwszą wykrytą twarz
                x, y, w, h = faces[0]
                face_roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                break
            else:
                print("❌ Nie wykryto twarzy na obrazie! Spróbuj ponownie.")
        elif key == 27:  # ESC przerywa
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return face_roi


def build_face_model():
    """Wczytuje zapisane zdjęcia i trenuje model Eigenfaces."""
    user_to_id, _ = get_user_mappings()
    images = []
    labels = []
    
    for username, uid in user_to_id.items():
        user_dir = os.path.join(BASE_DIR, username)
        for file in os.listdir(user_dir):
            if file.startswith("face_") and file.endswith(".jpg"):
                img_path = os.path.join(user_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Na wypadek gdyby rozmiary się nie zgadzały
                    img = cv2.resize(img, FACE_SIZE)
                    images.append(img)
                    labels.append(uid)
                    
    if len(images) == 0:
        return None
        
    # Tworzenie i trenowanie modelu Eigenfaces
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer


########################################
# Capture (Klawiatura)
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
# Registration (Multimodalna)
########################################

def collect_data(username):
    ensure_data_dir()

    user_dir = os.path.join(BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    # 1. KROK: Pobranie próbek twarzy
    print(f"\n🎯 Krok 1: Rejestracja Twarzy dla {username}")
    print("Zostanie pobranych 5 zdjęć twarzy.")
    for i in range(5):
        print(f"\n--- Zdjęcie twarzy {i + 1}/5 ---")
        face_img = capture_face_image()
        if face_img is not None:
            face_path = os.path.join(user_dir, f"face_{i + 1}.jpg")
            cv2.imwrite(face_path, face_img)
            print(f"✔ Zapisano zdjęcie: {face_path}")
        else:
            print("❌ Pominięto lub nie udało się zapisać zdjęcia.")

    # 2. KROK: Pobranie próbek klawiaturowych
    print(f"\n🎯 Krok 2: Rejestracja Klawiatury dla {username}")
    print("Każdy użytkownik powinien mieć co najmniej 5 próbek z różnych tekstów.")

    base_n = 1
    while os.path.exists(os.path.join(user_dir, f"{username}_{base_n}.csv")):
        base_n += 1

    for i in range(5):
        print(f"\n--- Próba klawiatury {i + 1}/5 ---")
        events = capture_events()

        file_path = os.path.join(user_dir, f"{username}_{base_n + i}.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            for t, key, event in events:
                f.write(f"{t};{key};{event}\n")

        print(f"✔ Zapisano dynamikę pisania: {file_path}")


########################################
# Multimodal Verification
########################################

def verify_user_multimodal():
    # Krok 1: Użytkownik podaje kim jest
    username = input("Podaj login użytkownika: ").strip()
    if not username:
        print("\n❌ Nie podano użytkownika.")
        return

    user_to_id, _ = get_user_mappings()
    if username not in user_to_id:
        print(f"\n❌ Użytkownik '{username}' nie istnieje w bazie danych.")
        return

    print(f"\nRozpoczynam weryfikację dla użytkownika: {username}")
    print("="*40)

    # Krok 2: Pobranie obrazu twarzy
    print("\n📸 [MODUŁ TWARZY] Przygotowanie do skanowania twarzy...")
    face_img = capture_face_image()
    if face_img is None:
        print("❌ Nie przechwycono obrazu twarzy. Przerywam proces.")
        return

    # Krok 3: Pobranie próby klawiaturowej
    print("\n⌨️ [MODUŁ KLAWIATURY] Przygotowanie do pisania tekstu...")
    vocab = get_vocab()
    keystroke_sample = capture_sample(vocab)

    # Parametry do analizy klawiatury
    k, metric, threshold_input = choose_parameters()
    
    # --- PROCES ANALIZY ---

    # 1. Ocena modułu twarzy
    face_model = build_face_model()
    face_verified = False
    face_confidence = float('inf')
    
    if face_model is None:
        print("\n⚠️ Brak wytrenowanego modelu twarzy w bazie.")
    else:
        label_id, face_confidence = face_model.predict(face_img)
        expected_id = user_to_id[username]
        
        # W algorytmie Eigenfaces confidence to odległość Euklidesowa (im mniejsza, tym lepiej)
        if label_id == expected_id and face_confidence < DEFAULT_FACE_THRESHOLD:
            face_verified = True

    # 2. Ocena modułu klawiatury
    keystroke_model = build_model(BASE_DIR, vocab)
    if len(keystroke_model["X"]) == 0:
        print("\n❌ Brak danych klawiaturowych w bazie.")
        return

    keystroke_threshold = threshold_input
    if keystroke_threshold is None:
        keystroke_threshold = estimate_threshold(keystroke_model, k=k, metric=metric, percentile=97.5, margin=1.15)

    keystroke_verified, kb_nearest_dist, kb_avg_dist = verify(
        keystroke_model,
        keystroke_sample,
        username,
        k=k,
        metric=metric,
        threshold=keystroke_threshold,
    )

    # Krok 4: Kombinacja cech i werdykt (Fuzja na poziomie decyzji - AND)
    print("\n" + "="*40)
    print("📊 PODSUMOWANIE ANALIZY:")
    print(f" -> Twarz:      {'ZGODNA' if face_verified else 'NIEZGODNA'} (Dystans: {face_confidence:.2f}, Próg: {DEFAULT_FACE_THRESHOLD})")
    print(f" -> Klawiatura: {'ZGODNA' if keystroke_verified else 'NIEZGODNA'} (Najbliższy dystans: {kb_nearest_dist:.4f}, Próg: {keystroke_threshold:.4f})")
    print("="*40)

    # Strategia fuzji: Pełna zgodność obu systemów (AND)
    if face_verified and keystroke_verified:
        print(f"\n✅ [WERDYKT] DOSTĘP PRZYZNANY. Tożsamość użytkownika '{username}' potwierdzona pomyślnie dwoma kanałami.")
    else:
        print(f"\n❌ [WERDYKT] DOSTĘP ODRZUCONY. Co najmniej jeden moduł biometryczny zgłosił błąd dopasowania.")


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
    print("\n=== MULTIMODALNY SYSTEM BIOMETRYCZNY ===")
    print("1. Rejestracja (Twarz + Klawiatura)")
    print("2. Multimodalna Weryfikacja użytkownika")
    print("3. Test modułu klawiatury + wykres")
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
            verify_user_multimodal()

        elif choice == "3":
            test_and_plot()

        elif choice == "0":
            break

        else:
            print("❌ Zła opcja")