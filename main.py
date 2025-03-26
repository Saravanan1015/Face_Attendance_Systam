import os
import datetime
import pickle
import subprocess
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        # Main buttons
        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login, fg='green')
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'Logout', 'red', self.logout, fg='red')
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
        self.main_window,'Register New User', 'gray', self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)
        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'


    def add_webcam(self, label):
        self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def login(self):
        """Handles user login with full name display."""
        recognized_name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if recognized_name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Oops...', 'Unknown user. Please Register Or Try Again.')
            return

        # Ensure full name is retrieved correctly from database
        full_name = self.get_full_name_from_db(recognized_name)

        util.msg_box('Welcome!', f'Welcome! , {full_name}.')
        with open(self.log_path, 'a') as f:
            f.write(f'{full_name},{datetime.datetime.now()},in\n')


    def logout(self):
        """Handles user logout with full name display."""
        recognized_name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if recognized_name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Oops...', 'Unknown User. Please Register Or Try Again.')
            return

        # Ensure full name is retrieved correctly from database
        full_name = self.get_full_name_from_db(recognized_name)

        util.msg_box('Thank You,See You Again!', f'Byee! , {full_name}.')
        with open(self.log_path, 'a') as f:
            f.write(f'{full_name},{datetime.datetime.now()},out\n')


    def register_new_user(self):
        """Opens user registration window."""
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        # Entry widget for username
        # Entry widget for username (increased width and font size)
        self.entry_text_register_new_user = tk.Entry(
        self.register_new_user_window, font=("Arial", 16), width=30
)
        self.entry_text_register_new_user.place(x=750, y=150, height=60)  # Adjusted height



        # Label
        tk.Label(self.register_new_user_window, text="Enter your name:", font=("Arial", 16)).place(x=750, y=100)

        # Accept and Try Again buttons
        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user , fg='green'
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user , fg='red'
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        # Webcam capture display
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)
    
    def try_again_register_new_user(self):
        """Closes the registration window."""
        self.register_new_user_window.destroy()

    def get_full_name_from_db(self, recognized_name):
        """Finds the full name of a recognized user based on stored files."""
        for file in os.listdir(self.db_dir):
            if file.endswith(".jpg") and recognized_name.lower() in file.lower():
                return file.replace(".jpg", "")  # Remove ".pickle" to get full name
        return recognized_name  # Fallback in case no match is found

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        print("Starting Tkinter main loop...")
        self.main_window.mainloop()
        print("Mainloop exited.")

    def accept_register_new_user(self):
        """Registers a new user with face embeddings."""
        name = self.entry_text_register_new_user.get().strip()
        if not name:
            util.msg_box("Error", "Name cannot be empty.")
            return

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if not embeddings:
            util.msg_box("Error", "No face detected. Try again.")
            return

        # Save the embeddings
        file_path = os.path.join(self.db_dir, f"{name}.jpg")
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings[0], file)

        util.msg_box('Success!', f'User {name} was registered successfully!')
        
        # Close registration window
        self.register_new_user_window.destroy()

        # Kill Python process after 5 seconds
        self.main_window.after(5000, self.execute_kill_command)


if __name__ == "__main__":
     app = App()
     app.start()
