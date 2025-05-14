import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
from PIL import Image, ImageTk, ImageChops
import os
from tkinter.font import Font
import cv2
import numpy as np
import matplotlib as plt
import ImgScript
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt


ctk.set_appearance_mode("light")

# ألوان جديدة
# #493628 dark brown
# #AB886D light brown
# #D6C0B3 beige
# #E4E0E1 light gray

class SimpleUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Imagino App")
        self.state('zoomed')
        self.configure(padx=20, pady=20, bg='#E4E0E1')
        self.iconbitmap('logo1.ico')

        self.loaded_images = []
        self.before_image_ref = None
        self.after_image_ref = None
        from PIL import ImageTk, Image
        self.transparent_img = ImageTk.PhotoImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))



        # Add modern app title
        self.app_title = ctk.CTkLabel(
            self,
            text="No Filter Selected Yet",
            font=ctk.CTkFont(family='Comic Sans MS',size=24, weight="bold"),
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
                    'MEDIAN', 'MODE', 'S&P AVG filter', 'S&P Median Filter', 'S&P Outlier filter', 'IMG AVG', 'GAUSSIAN AVG',
                    'IMG dilation','IMG erosion','IMG opening',
                    'BG_THR','AUTO_THR','ADAP_THR','Sobel DET']
        left_button_names = button_names[:15]  # First 10 names
        right_button_names = button_names[15:]  # Remaining 10 names

    # Create side boxes for the left and right sides
        self.create_side_boxes(content_frame, column=0, button_names=left_button_names)
        self.create_side_boxes(content_frame, column=2, button_names=right_button_names)

        # المربعات على الشمال
        #self.create_side_boxes(content_frame, column=0)

        # منطقة الصور
        image_section = ctk.CTkFrame(content_frame, fg_color='#AB886D', corner_radius=30, width=600, height=400)
        image_section.grid(row=0, column=1, sticky="nsew", padx=5, pady=20)
        image_section.grid_propagate(False)  # Prevent frame from resizing

        image_frame = ctk.CTkFrame(image_section, fg_color='#AB886D', corner_radius=20, width=580, height=350)
        image_frame.pack(pady=20)
        image_frame.pack_propagate(False)  # Prevent frame from resizing

        self.before_label = ctk.CTkLabel(
            image_frame, 
            text="Before", 
            width=300, 
            height=290,
            fg_color="#E4E0E1", 
            text_color="gray",
            font=('Comic Sans MS',25), 
            corner_radius=20
        )
        self.before_label.grid(row=0, column=0, padx=15, pady=60, sticky="nsew")
        self.before_label.drop_target_register(DND_FILES)
        self.before_label.dnd_bind("<<Drop>>", self.on_image_drop)

        self.after_label = ctk.CTkLabel(
            image_frame, 
            text="After", 
            width=300, 
            height=290,
            fg_color="#E4E0E1", 
            text_color="gray", 
            font=('Comic Sans MS',25), 
            corner_radius=20
        )
        self.after_label.grid(row=0, column=1, padx=15, pady=60, sticky="nsew")

        upload_btn = HoverButton(image_section, text="Upload Image", command=self.load_image)
        upload_btn.grid(row=0, column=0, padx=260, pady=15)

        reset_btn = HoverButton(image_section, text="Reset Images", command=self.reset_images)
        reset_btn.pack(pady=10)

        save_btn = HoverButton(image_section, text="Save Image", command=self.save_image)
        save_btn.pack(pady=15)

    
        self.image_paths = []  # To store the paths of the two uploaded images
        self.selected_operation = None  # To store the selected operation
        self.filtered_image = None
    def pathtkimage(path):
        imagepathed=Image.open(path)
        return ImageTk.PhotoImage(imagepathed)
    
    def save_image(self):
        if hasattr(self.after_label, 'image') and self.after_label.image:
            image_to_save = self.after_label.image

            if isinstance(image_to_save, ImageTk.PhotoImage):
                pil_image = ImageTk.getimage(image_to_save)  # تحويل PhotoImage إلى PIL.Image
            else:
                pil_image = image_to_save

            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])

            if file_path:
                pil_image.save(file_path)
                print(f"Image saved at {file_path}")
            else:
                print("Save operation cancelled.")
        else:
            print("No image to save.")

    
    def plot_histograms(self,hist_orig, hist_equalized):
        """
        Plot the original and equalized histograms in a new window.
        """
        plt.figure(figsize=(12, 6))

        # Plot the original histogram
        plt.subplot(1, 2, 1)
        plt.title("Original Histogram")
        plt.bar(range(256), hist_orig, color='gray')
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # Plot the equalized histogram
        plt.subplot(1, 2, 2)
        plt.title("Equalized Histogram")
        plt.bar(range(256), hist_equalized, color='gray')
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # Show the plots
        plt.tight_layout()
        plt.show()

    def plot_histogram_stretching(self,img, stretched_img):
        """
        Plot the histograms of the original and stretched images in a new window.
        """
        plt.figure(figsize=(12, 6))

        # Histogram of Original Image
        plt.subplot(1, 2, 1)
        plt.hist(img.ravel(), bins=256, color='black', alpha=0.7)
        plt.xlim([0, 256])
        plt.title('Histogram of Original Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        # Histogram of Stretched Image
        plt.subplot(1, 2, 2)
        plt.hist(stretched_img.ravel(), bins=256, color='black', alpha=0.7)
        plt.xlim([0, 256])
        plt.title('Histogram of Stretched Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        # Show the plots
        plt.tight_layout()
        plt.show()


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
                       
                        if name == 'Complement':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.complement(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                       
                        if name == 'Hist stretching':
                            
                            pilimg = Image.open(self.image_paths[0]).convert("L")  
                            npimg = np.array(pilimg)
                            stretched_img = ImgScript.histogram_stretching(npimg)
                            hist_orig, _ = np.histogram(npimg.flatten(), bins=256, range=(0, 256))
                            hist_stretched, _ = np.histogram(stretched_img.flatten(), bins=256, range=(0, 256))
                            resultpil = Image.fromarray(stretched_img)
                            self.display_image(resultpil, label=self.after_label)
                            self.plot_histogram_stretching(npimg, stretched_img)
                    
                        
                        if name == 'Hist equalization':

                            pilimg = Image.open(self.image_paths[0]).convert("L")  
                            npimg = np.array(pilimg)
                            equalized_img, hist_orig, hist_equalized = ImgScript.histogram_equalization(npimg)
                            resultpil = Image.fromarray(equalized_img)
                            self.display_image(resultpil, label=self.after_label)
                            self.plot_histograms(hist_orig, hist_equalized)
                        
                        if name == 'Laplacian filter':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.laplacian_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                    
                        if name == 'Change color':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.change_red_lighting(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)

                        if name == 'Eliminate color':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.eliminate_red(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                       
                        if name == 'Swap channel':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.swap_r_to_g(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)

                        if name == 'MAX':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.maximum_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                            # ImgScript.maximum_filter(SimpleUI.pathtkimage(self.image_paths[0]))
                       
                        if name == 'MIN':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.minimum_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                            # ImgScript.minimum_filter(SimpleUI.pathtkimage(self.image_paths[0]))
                                    
                        if name == 'MEDIAN':
                            # image1 = Image.open(path1).convert("RGB")
                            # ImgScript.median_filter(SimpleUI.pathtkimage(self.image_paths[0]))
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.median_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                            
                        if name == 'MODE':
                            # image1 = Image.open(path1).convert("RGB")
                            # ImgScript.median_filter(SimpleUI.pathtkimage(self.image_paths[0]))
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.mode_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                            

                        if name == 'AVG filter':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.low_pass_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)
                        
                        if name == 'S&P AVG filter':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.salt_pepper_avg(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)

                        if name == 'S&P Median Filter':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0])
                            npimg=np.array(pilimg)
                            result_image = ImgScript.salt_pepper_median(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label) 


                        if name == 'S&P Outlier filter':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.salt_pepper_outlier(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label)


                        if name == 'IMG AVG':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.gaussian_image_averaging(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label) 


                        if name == 'GAUSSIAN AVG':
                            # print(self.loaded_images[0])
                            pilimg=Image.open(self.image_paths[0]).convert("RGB")
                            npimg=np.array(pilimg)
                            result_image = ImgScript.gaussian_average_filter(npimg)
                            print(result_image.shape)
                            resultpil=Image.fromarray(result_image)
                            self.display_image(resultpil, label=self.after_label) 


                        if name == 'BG_THR':
                            # Open the image
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)

                            # Call the basic_global_thresholding function from ImgScript
                            thresholded_img = ImgScript.basic_global_thresholding(npimg)

                            # Convert the result back to a PIL image
                            resultpil = Image.fromarray(thresholded_img)

                            # Display the result in the "After" label
                            self.display_image(resultpil, label=self.after_label) 


                        if name == 'AUTO_THR':
                            # Open the image and convert it to grayscale
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)

                            # Call the automatic_thresholding function from ImgScript
                            thresholded_img = ImgScript.automatic_thresholding(npimg)

                            # Convert the result back to a PIL image
                            resultpil = Image.fromarray(thresholded_img)

                            # Display the result in the "After" label
                            self.display_image(resultpil, label=self.after_label)


                        if name == 'ADAP_THR':
                            # Open the image and convert it to grayscale
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)

                            # Call the adaptive_thresholding function from ImgScript
                            thresholded_img = ImgScript.adaptive_thresholding(npimg)

                            # Convert the result back to a PIL image
                            resultpil = Image.fromarray(thresholded_img)

                            # Display the result in the "After" label
                            self.display_image(resultpil, label=self.after_label)


                        if name == 'Sobel DET':
                            # Open the image and convert it to grayscale
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)

                            # Call the sobel_edge_detection function from ImgScript
                            sobel_img = ImgScript.sobel_edge_detection(npimg)

                            # Convert the result back to a PIL image
                            resultpil = Image.fromarray(sobel_img)

                            # Display the result in the "After" label
                            self.display_image(resultpil, label=self.after_label)


                        if name == 'IMG dilation':
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)
                            result_img = ImgScript.manual_dilation(npimg)  # Call the manual_dilation function
                            resultpil = Image.fromarray((result_img * 255).astype(np.uint8))  # Convert binary to grayscale
                            self.display_image(resultpil, label=self.after_label)   


                        if name == 'IMG erosion':
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)
                            result_img = ImgScript.manual_erosion(npimg)  # Call the manual_erosion function
                            resultpil = Image.fromarray((result_img * 255).astype(np.uint8))  # Convert binary to grayscale
                            self.display_image(resultpil, label=self.after_label)    


                        if name == 'IMG opening':
                            pilimg = Image.open(self.image_paths[0]).convert("L")  # Convert to grayscale
                            npimg = np.array(pilimg)
                            result_img = ImgScript.manual_opening(npimg)  # Call the manual_opening function
                            resultpil = Image.fromarray((result_img * 255).astype(np.uint8))  # Convert binary to grayscale
                            self.display_image(resultpil, label=self.after_label)         

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

# def display_image(self, file_path, label):
#     image = Image.open(file_path)
#     image = image.resize((200, 200))  # Adjust size as needed
#     photo = ImageTk.PhotoImage(image)

#     label.configure(image=photo, text="")  # Remove text, show image
#     label.image = photo  # Keep reference to avoid garbage collection
   

    def on_image_drop(self, event):
        path = event.data.strip("{").strip("}")
        if os.path.isfile(path):
            self.display_image(path)

    def display_image(self, source, label):
        try:
            if isinstance(source, str):
                image = Image.open(source).convert("RGB")
            else:
                image = source  # Already a PIL.Image
                

            target_width, target_height = 260, 280
            img_width, img_height = image.size

            target_ratio = target_width / target_height
            img_ratio = img_width / img_height

            if img_ratio > target_ratio:
                new_height = target_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = target_width
                new_height = int(new_width / img_ratio)

            image = image.resize((new_width, new_height), Image.LANCZOS)

            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            image = image.crop((left, top, right, bottom))

            loaded_image = ImageTk.PhotoImage(image)

            # تأجيل تعيين الصورة لحين تأكيد أن الـ label موجود فعليًا في واجهة Tkinter
            def update_label():
                label.configure(image=loaded_image, text="")
                label.image = loaded_image

                # تخزين مرجعية منفصلة لكل label (حل نهائي)
                if label == self.before_label:
                    self.before_image_ref = loaded_image
                elif label == self.after_label:
                    self.after_image_ref = loaded_image

            self.after(10, update_label)  # شغل بعد 10ms للتأكد من التثبيت في الواجهة

        except Exception as e:
            print(f"Error displaying image: {e}")
            label.configure(text="Error loading image", image=None)
            label.image = None





 
    def perform_operation(self, path1, path2, operation):
        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")

        image1 = image1.resize((300, 290))
        image2 = image2.resize((300, 290))

        # Convert images to NumPy arrays
        npimg1 = np.array(image1)
        npimg2 = np.array(image2)

        if operation == "ADD":
            # Call the addition function from ImgScript
            result_image = ImgScript.addition(npimg1, npimg2)

            # Convert the result back to a PIL image
            resultpil = Image.fromarray(result_image)

            # Display the result in the "After" label
            self.display_image(resultpil, label=self.after_label)
            
        elif operation == "SUB":
            
            result_image = ImgScript.subtraction(npimg1, npimg2)
            resultpil = Image.fromarray(result_image)
            self.display_image(resultpil, label=self.after_label)

        elif operation == "Division":
            
            result_image = ImgScript.division(npimg1, npimg2)
            resultpil = Image.fromarray(result_image)
            self.display_image(resultpil, label=self.after_label)

        else:
            raise ValueError(f"Unknown operation: {operation}")

        #return result  # NOTE: return image object not path


    
    def reset_images(self):
        self.image_paths = []
        # self.loaded_images.clear()
        self.filtered_image = None

        # Force-clear image using transparent 1x1 image
        self.before_label.configure(image=self.transparent_img, text="Before")
        self.before_label.image = self.transparent_img
        self.before_image_ref = None

        self.after_label.configure(image=self.transparent_img, text="After")
        self.after_label.image = self.transparent_img
        self.after_image_ref = None

        self.before_label.drop_target_register(DND_FILES)
        self.before_label.dnd_bind("<<Drop>>", self.on_image_drop)

        self.selected_operation = None

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