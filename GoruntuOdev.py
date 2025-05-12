import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🗸 Görüntü İşleme Arayüzü")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.history = []
        self.future = []   

        self.image = None
        

        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True)

        
        canvas = tk.Canvas(main_frame, bg="#f0f0f0", width=260, highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")

        
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.grid(row=0, column=2, padx=10, pady=20)
        

        title = tk.Label(scrollable_frame, text="🎨 Görüntü İşleme Aracı", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=10)

        self.points = []  
        self.selecting_points = False  


        
        buttons = [
            ("📂 Görsel Aç", self.open_image),
            ("🖤 Griye Çevir", self.convert_gray),
            ("🌈 RGB Kanalları", self.show_channels),
            ("🔄 Negatif", self.negative_image),
            ("☀️ Parlaklık +", self.increase_brightness),
            ("🌑 Parlaklık -", self.decrease_brightness),
            ("⛔ Eşikleme", self.threshold_image),
            ("📊 Histogram Göster", self.show_histogram),
            ("⚖️ Histogram Eşitle", self.equalize_histogram),
            ("🎚️ Kontrast Artır", self.increase_contrast),
            ("➡️ Taşı (Sağa)", self.translate_image),
            ("🪞 Aynalama (Yatay)", self.flip_image),
            ("🪟 Eğme (Shear)", self.shear_image),
            
            ("🔁 Döndür", self.rotate_image),
            ("✂️ Kırp", self.crop_image),
            ("📐 Perspektif Düzelt", self.start_perspective_correction),
            ("🔲 Ortalama Filtre", self.mean_filter),
            ("🧮 Medyan Filtre", self.median_filter),
            ("🧊 Gauss Filtre", self.gaussian_filter),
            ("🧯 Konservatif Filtre", self.conservative_filter),
            ("🌌 Crimmins Speckle", self.crimmins_filter),
            ("📏 Sobel", self.sobel_filter),
            ("📏 Prewitt", self.prewitt_filter),
            ("📏 Roberts", self.roberts_filter),
            ("📏 Compass", self.compass_filter),
            ("🔪 Canny", self.canny_edge),
            ("📈 Laplace", self.laplace_filter),
            ("🌊 Gabor", self.gabor_filter),
            ("🧭 Hough", self.hough_transform),
            ("🎯 k-means", self.kmeans_segmentation),
            ("🦴 Erode (Aşındır)", self.erode_image),
            ("🌕 Dilate (Genişlet)", self.dilate_image),
            ("💾 Fotoğrafı Kaydet", self.save_image),





        ]

        for text, cmd in buttons:
            self.make_button(scrollable_frame, text, cmd)

        
        self.label = tk.Label(right_frame, bg="#ffffff", relief="solid", bd=2, width=500, height=500)
        
        zoom_frame = tk.Frame(right_frame, bg="#f0f0f0")
        zoom_frame.pack(pady=10)

        tk.Label(zoom_frame, text="🔍 Yakınlaştırma (%):", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(side="left")

        self.zoom_scale = tk.Scale(zoom_frame, from_=10, to=300, orient="horizontal", length=200,
                                command=self.apply_zoom, bg="#f0f0f0")
        self.zoom_scale.set(100)
        self.zoom_scale.pack(side="left")

        self.label.pack()
        reset_btn = tk.Button(right_frame, text="♻️ Sıfırla (Orijinal)", command=self.reset_image,
        
        width=30, height=2, bg="#dc3545", fg="white", font=("Arial", 11, "bold"))
        reset_btn.pack(pady=10)

        btn_style = {"width": 12, "height": 1, "bg": "#6c757d", "fg": "white", "font": ("Arial", 10, "bold")}

        undo_btn = tk.Button(right_frame, text="↩️ Geri Al", command=self.undo, **btn_style)
        undo_btn.pack(pady=(0, 4))

        redo_btn = tk.Button(right_frame, text="↪️ İleri Al", command=self.redo, **btn_style)
        redo_btn.pack(pady=(0, 10))



        self.original_image = None  


    def make_button(self, parent, text, command):
        btn = tk.Button(parent, text=text, command=command, width=20, height=2,
                        bg="#007acc", fg="white", font=("Arial", 11, "bold"))
        btn.pack(pady=5)

    def open_image(self):
        path = filedialog.askopenfilename(title="Görsel Seç", filetypes=[("Görsel Dosyaları", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            print("⚠️ Hiçbir dosya seçilmedi.")
            return
        try:
            pil_img = Image.open(path).convert("RGB")
            self.original_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self.image = self.original_image.copy()
            self.display_image(self.image)
        except Exception as e:
            print(" Görsel okunamadı:", e)


    def display_image(self, img):
        if img is None:
            return

        try:
            zoom_factor = int(self.zoom_scale.get()) / 100.0
        except:
            zoom_factor = 1.0

        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        h, w = img.shape[:2]
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        
        im_pil = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.label.config(image=imgtk)
        self.label.image = imgtk



    def convert_gray(self):
        if self.image is not None:
            self.push_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray
            self.display_image(gray)

    def show_channels(self):
        if self.image is not None and len(self.image.shape) == 3:
            b, g, r = cv2.split(self.image)

            
            zeros = np.zeros_like(b)

           
            red_img = cv2.merge([zeros, zeros, r])
            green_img = cv2.merge([zeros, g, zeros])
            blue_img = cv2.merge([b, zeros, zeros])

            
            cv2.imshow("🔴 Kırmızı Kanal", red_img)
            cv2.imshow("🟢 Yeşil Kanal", green_img)
            cv2.imshow("🔵 Mavi Kanal", blue_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def negative_image(self):
        if self.image is not None:
            self.push_history()
            result = 255 - self.image
            self.image = result
            self.display_image(self.image)



    def increase_brightness(self):
        if self.image is not None:
            self.push_history()
            result = cv2.convertScaleAbs(self.image, alpha=1, beta=40)
            self.image = result
            self.display_image(self.image)


    def decrease_brightness(self):
        if self.image is not None:
            self.push_history()
            result = cv2.convertScaleAbs(self.image, alpha=1, beta=-40)
            self.image = result
            self.display_image(self.image)

    def threshold_image(self):
        if self.image is not None:
            self.push_history()
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.image = result
            self.display_image(self.image)
    
    def show_histogram(self):
        if self.image is not None:
            import matplotlib.pyplot as plt
            plt.figure()
            color = ('b', 'g', 'r') if len(self.image.shape) == 3 else ('k',)
            channels = [self.image] if len(self.image.shape) == 2 else cv2.split(self.image)
            for ch, col in zip(channels, color):
                hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.title("Histogram")
            plt.xlabel("Piksel Değeri")
            plt.ylabel("Sıklık")
            plt.show()

    def equalize_histogram(self):
        if self.image is not None:
            self.push_history()
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            result = cv2.equalizeHist(gray)
            self.image = result
            self.display_image(self.image)


    def increase_contrast(self):
        if self.image is not None:
            self.push_history()
            result = cv2.convertScaleAbs(self.image, alpha=1.5, beta=0)
            self.image = result
            self.display_image(self.image)

    def translate_image(self):
        if self.image is not None:
            self.push_history()
            rows, cols = self.image.shape[:2]
            M = np.float32([[1, 0, 50], [0, 1, 30]])
            result = cv2.warpAffine(self.image, M, (cols, rows))
            self.image = result
            self.display_image(self.image)

    def flip_image(self):
        if self.image is not None:
            self.push_history()
            result = cv2.flip(self.image, 1)
            self.image = result
            self.display_image(self.image)

    def shear_image(self):
        if self.image is not None:
            self.push_history()
            rows, cols = self.image.shape[:2]
            M = np.float32([[1, 0.5, 0], [0.2, 1, 0]])
            result = cv2.warpAffine(self.image, M, (int(cols * 1.5), int(rows * 1.5)))
            self.image = result
            self.display_image(self.image)
    


    def rotate_image(self):
        if self.image is not None:
            self.push_history()
            rows, cols = self.image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
            result = cv2.warpAffine(self.image, M, (cols, rows))
            self.image = result
            self.display_image(self.image)

    def crop_image(self):
        if self.image is not None:
            self.push_history()
            rows, cols = self.image.shape[:2]
            result = self.image[int(rows*0.25):int(rows*0.75), int(cols*0.25):int(cols*0.75)]
            self.image = result
            self.display_image(self.image)

    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.display_image(self.image)
    
    def start_perspective_correction(self):
        if self.image is not None:
            self.points = []
            self.selecting_points = True
            clone = self.image.copy()
            cv2.namedWindow("Nokta Seçimi")
            cv2.setMouseCallback("Nokta Seçimi", self.mouse_callback)
            while self.selecting_points:
                temp = clone.copy()
                for point in self.points:
                    cv2.circle(temp, point, 5, (0, 0, 255), -1)
                cv2.imshow("Nokta Seçimi", temp)
                if cv2.waitKey(1) & 0xFF == 27:  
                    break
            cv2.destroyWindow("Nokta Seçimi")

    def mouse_callback(self, event, x, y, flags, param):
        if self.selecting_points and event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) == 4:
                self.selecting_points = False
                self.apply_perspective_transform()
    def apply_perspective_transform(self):
        self.push_history()
        pts1 = np.float32(self.points)
        width = 400
        height = 400
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(self.image, M, (width, height))
        self.image = warped
        self.display_image(self.image)

    def mean_filter(self):
        if self.image is not None:
            self.push_history()
            result = cv2.blur(self.image, (5, 5))
            self.image = result
            self.display_image(result)


    def median_filter(self):
        if self.image is not None:
            self.push_history()
            result = cv2.medianBlur(self.image, 5)
            self.image = result
            self.display_image(result)


    def gaussian_filter(self):
        if self.image is not None:
            self.push_history()
            result = cv2.GaussianBlur(self.image, (5, 5), 0)
            self.image = result
            self.display_image(result)


    def conservative_filter(self):
        if self.image is not None:
            img = self.image.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = img.copy()
            h, w = img.shape
            for y in range(1, h-1):
                for x in range(1, w-1):
                    local = img[y-1:y+2, x-1:x+2].flatten()
                    local = np.delete(local, 4)
                    min_val = np.min(local)
                    max_val = np.max(local)
                    if img[y, x] < min_val:
                        result[y, x] = min_val
                    elif img[y, x] > max_val:
                        result[y, x] = max_val
            self.image = result
            self.display_image(result)


    def crimmins_filter(self):
        if self.image is not None:
            img = self.image.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = img.copy()
            for i in range(2):
                for y in range(1, img.shape[0] - 1):
                    for x in range(1, img.shape[1] - 1):
                        p = img[y, x]
                        neighbors = [
                            img[y-1, x], img[y+1, x],
                            img[y, x-1], img[y, x+1],
                            img[y-1, x-1], img[y-1, x+1],
                            img[y+1, x-1], img[y+1, x+1]
                        ]
                        mean_n = np.mean(neighbors)
                        if p < mean_n:
                            result[y, x] += 1
                for y in range(1, img.shape[0] - 1):
                    for x in range(1, img.shape[1] - 1):
                        p = img[y, x]
                        neighbors = [
                            img[y-1, x], img[y+1, x],
                            img[y, x-1], img[y, x+1],
                            img[y-1, x-1], img[y-1, x+1],
                            img[y+1, x-1], img[y+1, x+1]
                        ]
                        mean_n = np.mean(neighbors)
                        if p > mean_n:
                            result[y, x] -= 1
            self.image = result
            self.display_image(result)

    def sobel_filter(self):
        if self.image is not None:
            self.push_history()
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result = cv2.magnitude(sobelx, sobely)
            result = cv2.convertScaleAbs(result)
            self.image = result
            self.display_image(self.image)

    def prewitt_filter(self):
        if self.image is not None:
            self.push_history()
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            x = cv2.filter2D(gray, -1, kernelx)
            y = cv2.filter2D(gray, -1, kernely)
            result = cv2.magnitude(np.float32(x), np.float32(y))
            result = cv2.convertScaleAbs(result)
            self.image = result
            self.display_image(self.image)

    def roberts_filter(self):
        if self.image is not None:
            self.push_history()
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            x = cv2.filter2D(gray, -1, kernelx)
            y = cv2.filter2D(gray, -1, kernely)
            result = cv2.magnitude(np.float32(x), np.float32(y))
            result = cv2.convertScaleAbs(result)
            self.image = result
            self.display_image(self.image)

    def compass_filter(self):
        if self.image is not None:
            self.push_history()
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernels = [
                np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  
                np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  
                np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  
                np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  
                np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  
                np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  
                np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  
                np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])   
            ]
            responses = [cv2.filter2D(gray, -1, k) for k in kernels]
            result = np.max(np.stack(responses), axis=0)
            self.image = result
            self.display_image(self.image)

    def canny_edge(self):
        if self.image is not None:
            self.push_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            edges = cv2.Canny(gray, 100, 200)
            self.image = edges
            self.display_image(self.image)

    def laplace_filter(self):
        if self.image is not None:
            self.push_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            self.image = laplacian
            self.display_image(self.image)

    def gabor_filter(self):
        if self.image is not None:
            self.push_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            self.image = filtered
            self.display_image(self.image)

    def hough_transform(self):
        if self.image is not None:
            self.push_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for rho, theta in lines[:,0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 1)
            self.image = result
            self.display_image(self.image)

    def kmeans_segmentation(self):
        if self.image is not None:
            self.push_history()
            Z = self.image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 4
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented = centers[labels.flatten()].reshape(self.image.shape)
            self.image = segmented
            self.display_image(self.image)


    def erode_image(self):
        if self.image is not None:
            self.push_history()
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.erode(self.image, kernel, iterations=1)
            self.image = result
            self.display_image(self.image)


    def dilate_image(self):
        if self.image is not None:
            self.push_history()
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.dilate(self.image, kernel, iterations=1)
            self.image = result
            self.display_image(self.image)





    def apply_zoom(self, value):
        self.display_image(self.image)




        try:
            zoom_factor = int(value) / 100.0
            height, width = self.image.shape[:2]
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            resized = cv2.resize(self.image, new_size, interpolation=cv2.INTER_LINEAR)
            self.display_image(resized)
        except Exception as e:
            print("Zoom hatası:", e)

    def push_history(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.future.clear()

    def undo(self):
        if self.history:
            self.future.append(self.image.copy())
            self.image = self.history.pop()
            self.display_image(self.image)

    def redo(self):
        if self.future:
            self.history.append(self.image.copy())
            self.image = self.future.pop()
            self.display_image(self.image)

    def save_image(self):
        if self.image is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG dosyası", "*.png"), ("JPEG dosyası", "*.jpg"), ("Tüm dosyalar", "*.*")],
                title="Görseli Kaydet"
            )
            if path:
                try:
                    cv2.imwrite(path, self.image)
                    print("✅ Görsel başarıyla kaydedildi:", path)
                except Exception as e:
                    print("❌ Kaydetme hatası:", e)
                    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()