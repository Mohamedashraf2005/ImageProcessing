import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
from PIL import Image, ImageTk
import os

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# خلي بالك لازم نرث من TkinterDnD.Tk علشان السحب
class SimpleUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laplacian Filter UI (CustomTkinter)")
        self.geometry("1000x600")
        self.configure(padx=20, pady=20)

        self.loaded_image = None

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True)

        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        content_frame.columnconfigure(0, weight=2)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)  # Allow row to expand

        left = ctk.CTkFrame(content_frame)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        filter_title = ctk.CTkLabel(
            left,
            text="No Filter Selected",  # Proper title
            fg_color="#2a3bff",
            text_color="white",
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10,
            width=220,
            height=40
        )
        filter_title.pack(pady=(10, 20))

        image_frame = ctk.CTkFrame(left)
        image_frame.pack(pady=10)

        self.before_label = ctk.CTkLabel(image_frame, text="Before", width=350, height=300, fg_color="#d9d9d9", text_color="black")
        self.before_label.grid(row=0, column=0, padx=25)
        self.before_label.drop_target_register(DND_FILES)
        self.before_label.dnd_bind("<<Drop>>", self.on_image_drop)

        self.after_label = ctk.CTkLabel(image_frame, text="After", width=350, height=300, fg_color="#d9d9d9", text_color="black")
        self.after_label.grid(row=0, column=1, padx=10)

        upload_btn = ctk.CTkButton(left, text="Upload", command=self.load_image, width=130)
        upload_btn.pack(pady=20)

        right = ctk.CTkFrame(content_frame)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        grid_frame = ctk.CTkFrame(right)
        grid_frame.pack(pady=10)

        for i in range(5):
            for j in range(3):
                box = ctk.CTkFrame(grid_frame, width=90, height=70, fg_color="#6a5acd", corner_radius=8)
                box.grid(row=i, column=j, padx=5, pady=5)
                # label = ctk.CTkLabel(box, text=f"{i},{j}", text_color="white")
                # label.place(relx=0.5, rely=0.5, anchor="center")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        self.display_image(file_path)

    def on_image_drop(self, event):
        path = event.data.strip("{").strip("}")
        if os.path.isfile(path):
            self.display_image(path)

    def display_image(self, path):
        image = Image.open(path).resize((350, 300))
        self.loaded_image = ImageTk.PhotoImage(image)
        self.before_label.configure(image=self.loaded_image, text="")
        self.before_label.image = self.loaded_image
        self.after_label.configure(text="Filtered Image\n(To Be Added)", fg_color="#d9d9d9", text_color="black")

if __name__ == "__main__":
    app = SimpleUI()
    app.mainloop()