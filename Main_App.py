import customtkinter as ctk
import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import Image, ImageTk
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class GestureApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.attributes("-fullscreen", True)

        bg_pil = Image.open("ML-BG.png").resize((self.winfo_screenwidth(), self.winfo_screenheight()))
        self.bg_image = ImageTk.PhotoImage(bg_pil)
        self.bg_label = ctk.CTkLabel(self, text="",image=self.bg_image)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load Model
        try:
            self.model = joblib.load('gesture_model.pkl')
        except:
            print("Model file not found!")

        # Setup MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        self.grid_columnconfigure(0, weight=1)
        
        # Video Display Label
        self.video_label = ctk.CTkLabel(self, text="", bg_color="transparent")
        self.video_label.place(relx=0.72, rely=0.47, anchor="center")

        # Prediction Label
        self.label_status = ctk.CTkLabel(self, text="Waiting...", font=("Montserrat", 65, "bold"), text_color="#db0d00", bg_color="#ffffff" )
        self.label_status.place(relx=0.06, rely=0.22, anchor="nw")

        # Create the Close Button
        self.close_button = ctk.CTkButton(
            self, 
            text="✕", 
            width=40, 
            height=40, 
            fg_color="#db0d00",      # Matches your red text color
            hover_color="#aa0a00",   # Darker red when hovering
            text_color="white",
            font=("Arial", 20, "bold"),
            corner_radius=0,         # Square look to match the corner
            command=self.on_closing # Calls the function to stop camera and exit
        )

        # Position it in the extreme top-right corner
        self.close_button.place(relx=1.0, rely=0.0, anchor="ne")

        self.update_frame()

    def on_closing(self):
        # Stop the camera 
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        # Close the window
        self.destroy()
        print("Program closed safely.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Flip and Process
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            left_hand_data = [0] * 42
            right_hand_data = [0] * 42
            prediction = "No Hand Detected"

            if results.multi_hand_landmarks:
                for i, hand_handedness in enumerate(results.multi_handedness):
                    side = hand_handedness.classification[0].label
                    hand_lms = results.multi_hand_landmarks[i]
                    
                    pts = []
                    for lm in hand_lms.landmark:
                        pts.extend([lm.x, lm.y])
                    
                    # Apply your "Right/Left Swap" fix
                    if side == "Left":
                        left_hand_data = pts
                    else:
                        right_hand_data = pts

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                # 2. Predict
                features = np.array([left_hand_data + right_hand_data])
                prediction = self.model.predict(features)[0]

            # 3. Update UI Text
            self.label_status.configure(text=f"{prediction}")

            # 4. Convert OpenCV image to Tkinter format
            # Resize slightly for better performance on Pi
            frame_small = cv2.resize(frame, (840, 690))
            img_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.configure(image=img_tk)
            self.video_label._image_cache = img_tk # Prevent garbage collection

        # 5. Schedule the next update (approx 30 FPS)
        self.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = GestureApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
