import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os
import threading
import json

class PlantIdentifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Philippine Medicinal Plants - AI-Powered Leaf Identification")
        self.root.geometry("800x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.camera = None
        self.camera_running = False
        self.interpreter = None
        self.class_names = []
        self.current_image = None
        
        # Plant information database
        self.plant_info = self.load_plant_info()
        
        # Setup UI
        self.setup_ui()
        
        # Load model
        self.load_model()
        
    def setup_ui(self):
        # Header (fixed at top)
        header_frame = tk.Frame(self.root, bg='#4CAF50', height=80)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="Philippine Medicinal Plants", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#4CAF50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="AI-Powered Leaf Identification", 
                                 font=('Arial', 12), fg='white', bg='#4CAF50')
        subtitle_label.pack()
        
        # Create scrollable area
        scroll_frame = tk.Frame(self.root, bg='#f0f0f0')
        scroll_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(scroll_frame, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f0f0f0')
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas (centered)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to canvas and configure centering
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        # Main content frame (now inside scrollable area) - centered with max width
        main_frame = tk.Frame(self.scrollable_frame, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=50)
        
        # Image display area (centered)
        self.image_frame = tk.Frame(main_frame, bg='#e8f5e8', relief='raised', bd=2)
        self.image_frame.pack(pady=(0, 20))
        
        # Placeholder for leaf icon (centered)
        self.image_label = tk.Label(self.image_frame, text="🍃", font=('Arial', 100), 
                                   bg='#e8f5e8', fg='#4CAF50')
        self.image_label.pack(pady=50, padx=100)
        
        # Button frame (centered)
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=(0, 20))
        
        # Create a sub-frame to center buttons with fixed width
        button_container = tk.Frame(button_frame, bg='#f0f0f0')
        button_container.pack()
        
        # Camera button
        self.camera_btn = tk.Button(button_container, text="📷 Camera", font=('Arial', 16, 'bold'),
                                   bg='#4CAF50', fg='white', relief='raised', bd=3,
                                   command=self.open_camera, cursor='hand2', width=12)
        self.camera_btn.pack(side='left', padx=(0, 10), pady=10, ipady=15)
        
        # Gallery button
        self.gallery_btn = tk.Button(button_container, text="🖼️ Gallery", font=('Arial', 16, 'bold'),
                                    bg='#FF9800', fg='white', relief='raised', bd=3,
                                    command=self.open_gallery, cursor='hand2', width=12)
        self.gallery_btn.pack(side='left', padx=(10, 0), pady=10, ipady=15)
        
        # Instructions frame (centered)
        info_frame = tk.Frame(main_frame, bg='#333333', relief='raised', bd=2)
        info_frame.pack(pady=(0, 20))
        
        info_title = tk.Label(info_frame, text="🤖 How AI Identification Works:", 
                             font=('Arial', 14, 'bold'), fg='#4CAF50', bg='#333333')
        info_title.pack(pady=(15, 5))
        
        instructions = [
            "1. Take or select a clear plant leaf photo",
            "2. AI automatically analyzes the leaf features", 
            "3. Get instant results with confidence score",
            "4. Learn about medicinal properties and uses"
        ]
        
        for instruction in instructions:
            label = tk.Label(info_frame, text=instruction, font=('Arial', 11), 
                           fg='#cccccc', bg='#333333')
            label.pack(pady=2, padx=30)
        
        tk.Label(info_frame, text="", bg='#333333').pack(pady=5)  # Spacer
        
        # Results frame (centered)
        self.results_frame = tk.Frame(main_frame, bg='#f0f0f0')
        self.results_frame.pack(expand=True, pady=20)
        
        # Camera frame (hidden initially)
        self.camera_frame = tk.Toplevel(self.root)
        self.camera_frame.withdraw()
        self.camera_frame.title("Camera")
        self.camera_frame.geometry("640x580")
        self.camera_frame.protocol("WM_DELETE_WINDOW", self.close_camera)
        
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(pady=10)
        
        camera_btn_frame = tk.Frame(self.camera_frame)
        camera_btn_frame.pack(pady=10)
        
        capture_btn = tk.Button(camera_btn_frame, text="📸 Capture", font=('Arial', 14, 'bold'),
                               bg='#4CAF50', fg='white', command=self.capture_image)
        capture_btn.pack(side='left', padx=10)
        
        close_btn = tk.Button(camera_btn_frame, text="❌ Close", font=('Arial', 14, 'bold'),
                             bg='#f44336', fg='white', command=self.close_camera)
        close_btn.pack(side='left', padx=10)
        
    def load_plant_info(self):
        """Load plant information database"""
        # This is a sample database - you can expand this with real medicinal information
        return {
            "Aloe_Vera": {
                "scientific_name": "Aloe barbadensis miller",
                "common_names": ["Aloe", "Sabila"],
                "medicinal_uses": ["Burns", "Skin healing", "Digestive aid"],
                "preparation": "Apply gel directly to skin or consume juice"
            },
            "Ginger": {
                "scientific_name": "Zingiber officinale",
                "common_names": ["Luya"],
                "medicinal_uses": ["Nausea", "Inflammation", "Digestive issues"],
                "preparation": "Tea, powder, or fresh root"
            },
            # Add more plants as needed based on your 40 classes
        }
    
    def load_model(self):
        """Load the TFLite model and class names"""
        try:
            # Load TFLite model
            model_path = "model.tflite"
            if os.path.exists(model_path):
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                
                # Load class names from dataset directory
                dataset_path = "Philippine Medicinal Plant Leaf Dataset"
                if os.path.exists(dataset_path):
                    self.class_names = sorted([name for name in os.listdir(dataset_path) 
                                             if os.path.isdir(os.path.join(dataset_path, name))])
                else:
                    # Fallback: define class names manually if dataset folder not found
                    self.class_names = [f"Class_{i}" for i in range(40)]
                
                messagebox.showinfo("Model Loaded", f"Successfully loaded model with {len(self.class_names)} classes")
            else:
                messagebox.showwarning("Model Not Found", "model.tflite not found. Please ensure the model file is in the same directory.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def open_camera(self):
        """Open camera window"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Camera Error", "Cannot access camera")
                return
            
            self.camera_running = True
            self.camera_frame.deiconify()
            self.update_camera()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to open camera: {str(e)}")
    
    def update_camera(self):
        """Update camera feed"""
        if self.camera_running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame for display
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update camera label
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo  # Keep a reference
                
            # Schedule next update
            self.camera_frame.after(10, self.update_camera)
    
    def capture_image(self):
        """Capture image from camera"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(frame_rgb)
                self.display_image(self.current_image)
                self.close_camera()
                self.predict_plant()
    
    def close_camera(self):
        """Close camera"""
        self.camera_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_frame.withdraw()
    
    def open_gallery(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select Plant Leaf Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                self.display_image(self.current_image)
                self.predict_plant()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, pil_image):
        """Display image in the main window"""
        # Resize image for display while maintaining aspect ratio
        display_image = pil_image.copy()
        display_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_image)
        
        # Update image label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
        
        # Update scroll region after image display
        self.root.after_idle(self._update_scroll_region)
    
    def predict_plant(self):
        """Predict plant species using the TFLite model"""
        if not self.interpreter or not self.current_image:
            return
        
        try:
            # Preprocess image
            img_resized = self.current_image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get input and output details
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Make prediction
            self.interpreter.set_tensor(input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(output_details[0]['index'])
            
            # Get results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = self.class_names[predicted_class_idx]
            
            # Display results
            self.display_results(predicted_class, confidence, predictions[0])
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict: {str(e)}")
    
    def display_results(self, predicted_class, confidence, all_predictions):
        """Display prediction results"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Results header (centered)
        results_header = tk.Label(self.results_frame, text="🔍 Identification Results", 
                                 font=('Arial', 18, 'bold'), fg='#2E7D32', bg='#f0f0f0')
        results_header.pack(pady=(20, 10))
        
        # Main result (centered with max width)
        result_frame = tk.Frame(self.results_frame, bg='white', relief='raised', bd=2)
        result_frame.pack(pady=10, padx=40)
        
        plant_name = tk.Label(result_frame, text=f"Plant: {predicted_class.replace('_', ' ')}", 
                             font=('Arial', 16, 'bold'), fg='#1B5E20', bg='white')
        plant_name.pack(pady=(15, 5))
        
        confidence_label = tk.Label(result_frame, text=f"Confidence: {confidence:.1f}%", 
                                   font=('Arial', 14), fg='#4CAF50', bg='white')
        confidence_label.pack(pady=(0, 15))
        
        # Progress bar for confidence (centered)
        progress_frame = tk.Frame(result_frame, bg='white')
        progress_frame.pack(pady=(0, 15))
        
        progress = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        progress['value'] = confidence
        progress.pack()
        
        # Plant information (if available) - centered
        if predicted_class in self.plant_info:
            info = self.plant_info[predicted_class]
            
            info_frame = tk.Frame(self.results_frame, bg='#E8F5E8', relief='raised', bd=2)
            info_frame.pack(pady=10, padx=40)
            
            info_title = tk.Label(info_frame, text="🌿 Medicinal Information", 
                                 font=('Arial', 14, 'bold'), fg='#2E7D32', bg='#E8F5E8')
            info_title.pack(pady=(10, 5))
            
            scientific_label = tk.Label(info_frame, text=f"Scientific Name: {info['scientific_name']}", 
                                       font=('Arial', 11), fg='#1B5E20', bg='#E8F5E8')
            scientific_label.pack(pady=2, padx=15)
            
            uses_label = tk.Label(info_frame, text=f"Uses: {', '.join(info['medicinal_uses'])}", 
                                 font=('Arial', 11), fg='#1B5E20', bg='#E8F5E8')
            uses_label.pack(pady=2, padx=15)
            
            prep_label = tk.Label(info_frame, text=f"Preparation: {info['preparation']}", 
                                 font=('Arial', 11), fg='#1B5E20', bg='#E8F5E8')
            prep_label.pack(pady=(2, 15), padx=15)
        
        # Top 3 predictions (centered)
        top3_frame = tk.Frame(self.results_frame, bg='#FFF3E0', relief='raised', bd=2)
        top3_frame.pack(pady=10, padx=40)
        
        top3_title = tk.Label(top3_frame, text="📊 Top 3 Predictions", 
                             font=('Arial', 14, 'bold'), fg='#E65100', bg='#FFF3E0')
        top3_title.pack(pady=(10, 5))
        
        # Get top 3 predictions
        top3_indices = np.argsort(all_predictions)[-3:][::-1]
        
        for i, idx in enumerate(top3_indices):
            class_name = self.class_names[idx].replace('_', ' ')
            conf = all_predictions[idx] * 100
            
            pred_label = tk.Label(top3_frame, text=f"{i+1}. {class_name}: {conf:.1f}%", 
                                 font=('Arial', 11), fg='#BF360C', bg='#FFF3E0')
            pred_label.pack(pady=2, padx=15)
        
        tk.Label(top3_frame, text="", bg='#FFF3E0').pack(pady=5)  # Spacer
        
        # Update scroll region after adding new content
        self.root.after_idle(self._update_scroll_region)
        
        # Update scroll region after adding content
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize to center content"""
        canvas_width = event.width
        frame_width = self.scrollable_frame.winfo_reqwidth()
        
        # Center the frame in the canvas
        if canvas_width > frame_width:
            x_offset = (canvas_width - frame_width) // 2
        else:
            x_offset = 0
            
        self.canvas.coords(self.canvas_window, x_offset, 0)
    
    def _update_scroll_region(self):
        """Update the scroll region to fit all content"""
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Re-center the content
        canvas_width = self.canvas.winfo_width()
        frame_width = self.scrollable_frame.winfo_reqwidth()
        
        if canvas_width > frame_width:
            x_offset = (canvas_width - frame_width) // 2
        else:
            x_offset = 0
            
        self.canvas.coords(self.canvas_window, x_offset, 0)

def main():
    root = tk.Tk()
    app = PlantIdentifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()