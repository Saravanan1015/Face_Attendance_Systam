import os
import pickle

import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    print(f"Creating button: {text} with command: {command}")  # Debugging print
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,  # This should be callable
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )
    return button




def get_img_label(window):
    label = tk.Label(window)
    #label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path):
    # it is assumed there will be at most 1 match in the db

    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted(os.listdir(db_path))

    match = False
    j = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])

        try:
            with open(path_, 'rb') as file:
                embeddings = pickle.load(file)
        except Exception as e:
            print(f"Error reading {path_}: {e}")
            j += 1
            continue

        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        j += 1

    if match:
        return db_dir[j - 1][:-7]  # Returns the name without '.pickle' extension
    else:
        return 'unknown_person'


def register_user(name, img, db_path):
    embeddings = face_recognition.face_encodings(img)
    if not embeddings:
        msg_box("Error", "No face detected. Please try again.")
        return
    
    embeddings = embeddings[0]

    try:
        os.makedirs(db_path, exist_ok=True)
        file_path = os.path.join(db_path, f'{name}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)
        msg_box('Success!', 'User was registered successfully!')
    except Exception as e:
        msg_box("Error", f"An error occurred during registration: {e}")
