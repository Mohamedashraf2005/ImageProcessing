import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# إعداد الشكل العام
ctk.set_appearance_mode("light")

# ألوان جديدة
PETROL_DARK = "#005f5f"  # بترولي غامق
WHITE = "#ffffff"        # أبيض
IMAGE_BG = "#f0f0f0"     # خلفية الصور
BOX_COLOR = PETROL_DARK  # نفس لون الأزرار

class SimpleUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer")
        self.geometry("1200x750")
        self.configure(padx=20, pady=20, bg=WHITE)

        self.loaded_image = None

        # Add modern app title
        app_title = ctk.CTkLabel(
            self,
            text="No Filter Selected Yet",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=PETROL_DARK,
            anchor="center"
        )
        app_title.pack(pady=(0, 0))  # Padding above and below the title

        # Main Frame
        main_frame = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=25)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Content Frame
        content_frame = ctk.CTkFrame(main_frame, fg_color=WHITE)
        content_frame.pack(fill="both", expand=True, padx=20, pady=120)

        content_frame.columnconfigure((0, 2), weight=1)
        content_frame.columnconfigure(1, weight=3)

        # المربعات على الشمال
        self.create_side_boxes(content_frame, column=0)

        # منطقة الصور
        image_section = ctk.CTkFrame(content_frame, fg_color=IMAGE_BG, corner_radius=30)
        image_section.grid(row=0, column=1, sticky="nsew", padx=20)

        image_frame = ctk.CTkFrame(image_section, fg_color=WHITE, corner_radius=20)
        image_frame.pack(pady=30)

        self.before_label = ctk.CTkLabel(image_frame, text="Before", width=330, height=280,
                                         fg_color="#eeeeee", text_color="gray", corner_radius=20)
        self.before_label.grid(row=0, column=0, padx=15)
        self.before_label.drop_target_register(DND_FILES)
        self.before_label.dnd_bind("<<Drop>>", self.on_image_drop)

        self.after_label = ctk.CTkLabel(image_frame, text="After", width=330, height=280,
                                        fg_color="#eeeeee", text_color="gray", corner_radius=20)
        self.after_label.grid(row=0, column=1, padx=15)

        upload_btn = HoverButton(image_section, text="Upload Image", command=self.load_image)
        upload_btn.pack(pady=10)

        # المربعات على اليمين
        self.create_side_boxes(content_frame, column=2)

    def create_side_boxes(self, parent, column):
        side_frame = ctk.CTkFrame(parent, fg_color=WHITE, corner_radius=25)
        side_frame.grid(row=0, column=column, padx=10, pady=10)

        grid_frame = ctk.CTkFrame(side_frame, fg_color=WHITE)
        grid_frame.pack(padx=10, pady=10)

        for i in range(5):
            for j in range(2):
                box = HoverBoxButton(grid_frame, text=f"Button {i*2+j+1}", command=lambda: print(f"Button {i*2+j+1} clicked"))
                box.grid(row=i, column=j, padx=5, pady=5)

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
        image = Image.open(path).resize((330, 280))
        self.loaded_image = ImageTk.PhotoImage(image)
        self.before_label.configure(image=self.loaded_image, text="")
        self.before_label.image = self.loaded_image

# زر بترولي غامق + Hover أبيض
class HoverButton(ctk.CTkButton):
    def __init__(self, master=None, **kwargs):
        self.normal_color = PETROL_DARK
        self.hover_color = WHITE
        self.normal_text = "white"
        self.hover_text = PETROL_DARK

        super().__init__(
            master,
            fg_color=self.normal_color,
            hover_color=self.hover_color,
            text_color=self.normal_text,
            corner_radius=20,
            font=ctk.CTkFont(size=14),
            width=160,
            height=40,
            **kwargs
        )

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.configure(fg_color=self.hover_color, text_color=self.hover_text)

    def on_leave(self, event):
        self.configure(fg_color=self.normal_color, text_color=self.normal_text)

# زر مربع مع تأثير Hover
class HoverBoxButton(ctk.CTkButton):
    def __init__(self, master=None, **kwargs):
        super().__init__(
            master,
            width=80,
            height=60,
            fg_color=BOX_COLOR,
            hover_color=WHITE,
            text_color="white",
            corner_radius=12,
            font=ctk.CTkFont(size=12),
            **kwargs
        )
        self.normal_color = BOX_COLOR
        self.hover_color = WHITE
        self.normal_text = "white"
        self.hover_text = PETROL_DARK

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.configure(fg_color=self.hover_color, text_color=self.hover_text)

    def on_leave(self, event):
        self.configure(fg_color=self.normal_color, text_color=self.normal_text)

# تشغيل التطبيق
if __name__ == "__main__":
    app = SimpleUI()
    app.mainloop()