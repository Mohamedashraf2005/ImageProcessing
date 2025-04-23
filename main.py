import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
from PIL import Image, ImageTk, ImageChops
import os
from tkinter.font import Font


ctk.set_appearance_mode("light")

# ألوان جديدة
# #493628 dark brown
# #AB886D light brown
# #D6C0B3 beige
# #E4E0E1 light gray

class SimpleUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing App")
        self.geometry("1300x750")
        self.configure(padx=20, pady=20, bg='#E4E0E1')
        self.iconbitmap('logo1.ico')

        self.loaded_image = None

        # Add modern app title
        self.app_title = ctk.CTkLabel(
            self,
            text="No Filter Selected Yet",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color='#493628',
            anchor="center"
        )
        self.app_title.pack(pady=(0,0))
        self.app_title.pack(pady=(0, 0))  # Padding above and below the title

        # Main Frame
        main_frame = ctk.CTkFrame(self, fg_color='#D6C0B3', corner_radius=25)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Content Frame
        content_frame = ctk.CTkFrame(main_frame, fg_color='#D6C0B3', corner_radius=25)
        content_frame.pack(fill="both", expand=True, padx=10, pady=60)

        content_frame.columnconfigure((0, 2), weight=1)
        content_frame.columnconfigure(1, weight=3)

        button_names = ['ADD', 'SUB', 'Division', 'Complement', 'Change color', 'Swap channel', 'Eliminate color',
                    'Hist stretching', 'Hist equalization', 'AVG filter', 'Laplacian filter', 'MAX', 'MIN',
                    'MEDIAN', 'MODE', 'S&P AVG filter', 'Median filter', 'Outlier filter', 'IMG AVG', 'GAUSSIAN AVG',
                    'INT boundary','EXT boundary','Morophology','IMG dilation','IMG erosion','IMG opening',
                    'BG_THR','AUTO_THR','ADAP_THR','Sobel DET']
        left_button_names = button_names[:15]  # First 10 names
        right_button_names = button_names[15:]  # Remaining 10 names

    # Create side boxes for the left and right sides
        self.create_side_boxes(content_frame, column=0, button_names=left_button_names)
        self.create_side_boxes(content_frame, column=2, button_names=right_button_names)

        # المربعات على الشمال
        #self.create_side_boxes(content_frame, column=0)

        # منطقة الصور
        image_section = ctk.CTkFrame(content_frame, fg_color='#AB886D', corner_radius=30,width=600, height=400)
        image_section.grid(row=0, column=1, sticky="nsew", padx=5,pady= 20)

        image_frame = ctk.CTkFrame(image_section, fg_color='#AB886D', corner_radius=20,width=580, height=350)
        image_frame.pack(pady=20)

        self.before_label = ctk.CTkLabel(image_frame, text="Before", width=300, height=290,
                                         fg_color="#E4E0E1", text_color="gray", corner_radius=20)
        self.before_label.grid(row=0, column=0, padx=5, pady=30)
        self.before_label.drop_target_register(DND_FILES)
        self.before_label.dnd_bind("<<Drop>>", self.on_image_drop)

        self.after_label = ctk.CTkLabel(image_frame, text="After", width=300, height=290,
                                        fg_color="#E4E0E1", text_color="gray", corner_radius=20)
        self.after_label.grid(row=0, column=1, padx=5,pady=30)

        upload_btn = HoverButton(image_section, text="Upload Image", command=self.load_image)
        upload_btn.pack(pady=20)

        reset_btn = HoverButton(image_section, text="Reset Images", command=self.reset_images)
        reset_btn.pack(pady=10)

    
        self.image_paths = []  # To store the paths of the two uploaded images
        self.selected_operation = None  # To store the selected operation


    def create_side_boxes(self, parent, column, button_names):
        side_frame = ctk.CTkFrame(parent, fg_color='#AB886D', corner_radius=25)
        side_frame.grid(row=0, column=column, padx=10, pady=10,sticky="nsew")

        grid_frame = ctk.CTkFrame(side_frame, fg_color='#AB886D', corner_radius=25)
        grid_frame.pack(padx=10, pady=10)

         # Calculate the width of the largest button name
        font = Font(family="TkDefaultFont", size=12)  # Use the font size of your buttons
        largest_name = max(button_names, key=len)
        button_width = font.measure(largest_name) + 40  # Add padding for aesthetic


        for i in range(8):
            for j in range(2):
                button_index = i * 2 + j
                if button_index >= len(button_names):  # Avoid index out of range
                   break
                button_name = button_names[button_index]

            # Define the command for each button
                def button_clicked(index=button_index, name=button_name):
                    print(f"Button {index + 1} clicked: {name}")
                    self.show_text(f"Applied {name}")  # Display text

                    #self.apply_filter(index + 1)  # Perform specific operation
                    if name in ["ADD", "SUB", "Division"]:
                        if len(self.image_paths) < 2:
                            print("Please upload two images first.")
                            self.show_text("Please upload two images first.")
                            return

                        # Perform the operation and display the result
                        result_image = self.perform_operation(self.image_paths[0], self.image_paths[1], operation=name)
                        self.display_image(result_image, label=self.after_label)

                        # Reset the image paths after applying the operation
                        self.image_paths = []
                        self.show_text("Operation applied. Reset to upload new images.")
                    else:
                        if len(self.image_paths) < 1:
                            print("Please upload at least one image first.")
                            self.show_text("Please upload at least one image first.")
                            return

                        #Apply a single-image filter (placeholder logic)
                        print(f"Applying filter '{name}' to the first image.")
                        self.display_image(self.image_paths[0], label=self.before_label)
                        self.show_text(f"Filter '{name}' applied to the first image.")
                            

            # Create the button with the updated name and command
                box = HoverBoxButton(grid_frame, text=button_name, command=lambda name=button_name, index=button_index: button_clicked(index, name),width=button_width,height= 55)
                box.grid(row=i, column=j, padx=5, pady=5)

    def load_image(self):

        if len(self.image_paths) >= 2:
            print("Two images are already uploaded. Please apply an operation or reset.")
            return

        # Allow the user to upload an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            print("No image selected.")
            return

        self.image_paths.append(file_path)  # Store the uploaded image path

        # Display the uploaded image in the appropriate label
        if len(self.image_paths) == 1:
            self.display_image(file_path, label=self.before_label)
        elif len(self.image_paths) == 2:
            self.display_image(file_path, label=self.after_label)

        # Update the app title to indicate readiness for an operation
        if len(self.image_paths) == 2:
            self.show_text("Images uploaded. Ready to apply an operation.")    

    def on_image_drop(self, event):
        path = event.data.strip("{").strip("}")
        if os.path.isfile(path):
            self.display_image(path)

    def display_image(self, path,label):
        try:
            image = Image.open(path).resize((330, 280))
            loaded_image = ImageTk.PhotoImage(image)
            label.configure(image=loaded_image, text="")
            label.image = loaded_image
        except Exception as e:
            print(f"Error displaying image: {e}")
 
    def perform_operation(self, path1, path2, operation):
    # Open the two images
        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")

    # Ensure both images are the same size
        image1 = image1.resize((330, 280))
        image2 = image2.resize((330, 280))

    # Perform the operation
        if operation == "ADD":
            result = Image.blend(image1, image2, alpha=0.5)  # Blend the two images
        elif operation == "SUB":
            result = ImageChops.subtract(image1, image2)  # Subtract the two images
        elif operation == "Division":
            # Perform division operation (example: pixel-wise division)
            result = ImageChops.darker(image1, image2)  # Placeholder for division logic
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Save the result to a temporary file
        result_path = "result_image.png"
        result.save(result_path)
        return result_path 
    
    def reset_images(self):
        # Clear the image paths
        self.image_paths = []

        # Reset the "Before" and "After" labels
        self.before_label.configure(image=None, text="Before")
        self.before_label.image = None
        self.after_label.configure(image=None, text="After")
        self.after_label.image = None

        # Reset the selected operation
        self.selected_operation = None

        # Update the app title or any other status
        self.show_text("Images reset. Ready to upload new ones.")
        print("Images have been reset.")



    def show_text(self, text):
        # Update the app title or any other label to display the text
        self.app_title.configure(text=text)

    def apply_filter(self, filter_number):
        # Perform a specific operation based on the filter number
        print(f"Applying filter {filter_number}")
        # Add your filter logic here   


# زر بترولي غامق + Hover أبيض
class HoverButton(ctk.CTkButton):
    def __init__(self, master=None,text="Button",command= None, **kwargs):
        self.normal_color = '#493628'
        self.hover_color = '#D6C0B3'
        self.normal_text = '#AB886D'
        self.hover_text = '#493628'

        super().__init__(
            master,
            fg_color=self.normal_color,
            hover_color=self.hover_color,
            text_color=self.normal_text,
            corner_radius=20,
            font=ctk.CTkFont(size=14),
            width=160,
            height=40,
            text= text,
            command=command,
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
            #width=80,
            #height=60,
            fg_color='#493628',
            hover_color='#D6C0B3',
            text_color='#AB886D',
            corner_radius=12,
            font=ctk.CTkFont(size=12),
            **kwargs
        )
        self.normal_color = '#493628'
        self.hover_color = '#D6C0B3'
        self.normal_text = '#AB886D'
        self.hover_text = '#493628'

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