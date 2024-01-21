from config.config import *
from model.model import *

Image.MAX_IMAGE_PIXELS = None

plt.rcParams['image.cmap'] = 'gray' 

class ImageProcessingApp():
    def __init__(self, root):
        # Initialize the application window
        self.root = root
        self.root.title("Entrenamiento")
        
         # Style configurations
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TEntry", padding=5, relief="flat", background="#eee")

        # Variables
        self.mean = tk.IntVar()
        self.std = tk.IntVar()
        self.rotate_limit = tk.IntVar()
        self.shift_limit = tk.DoubleVar()
        self.scale_limit = tk.DoubleVar()
        self.var_limit_min = tk.StringVar()
        self.var_limit_max = tk.StringVar()
        self.mask_fill_value = tk.IntVar()
        self.transpose_mask = tk.BooleanVar()

        # Variables
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.directory_path = tk.StringVar()
        self.size = tk.IntVar()
        self.valid = tk.BooleanVar()
        self.model_path = tk.StringVar()
        self.porcentaje = tk.StringVar()
        self.tarea = tk.StringVar()
        self.tarea1 = tk.StringVar()
        self.learning_rate = tk.DoubleVar()
        self.num_epochs = tk.IntVar()
        self.cluster = tk.IntVar()
        self.batch_size = tk.IntVar()
        self.num_carpetas_img = tk.IntVar()
        self.epoch_progress_var = tk.DoubleVar()
        self.loss_label_var = tk.StringVar()
        self.wandb_project_name = tk.StringVar()
        self.umbral = tk.IntVar()
        # Model and training configurations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        # Try loading values from the file
        self.cargar_valores_desde_archivo()

        # Augmentation lists for training and validation
        self.train_aug_list = [
            # A.RandomResizedCrop(
            #     size, size, scale=(0.85, 1.0)),
            A.Resize(64, 64),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.6),

            A.RandomBrightnessContrast(p=0.75),
            A.ShiftScaleRotate(rotate_limit=self.rotate_limit.get(),shift_limit=self.shift_limit.get(),scale_limit=self.scale_limit.get(),p=0.75),
            A.OneOf([
                    A.GaussNoise(var_limit=[int(self.var_limit_min.get()), int(self.var_limit_max.get())]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    ], p=0.4),
            # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=2, max_width=int(64 * 0.2), max_height=int(64 * 0.2), 
                            mask_fill_value=self.mask_fill_value.get(), p=0.5),
            # A.Cutout(max_h_size=int(size * 0.6),
            #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
            A.Normalize(
                mean= [0] * self.mean.get(),
                std= [1] * self.std.get(),
            ),
            ToTensorV2(transpose_mask=self.transpose_mask.get()),
        ]

        self.valid_aug_list = [
            A.Resize(self.size.get(), self.size.get()),
            A.Normalize(
                mean= [0] * self.mean.get(),
                std= [1] * self.std.get()
            ),
            ToTensorV2(transpose_mask=self.transpose_mask.get()),
        ]

        # Main frame setup
        main_frame = ttk.Frame(root, padding=(20, 10))
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Image Directory Section
        ttk.Label(main_frame, text="Directorio de Imágenes:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.directory_entry = ttk.Entry(main_frame, textvariable=self.directory_path)
        self.directory_entry.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Seleccionar Directorio", command=self.browse_directory, style="TButton").grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)

        # Model Directory Section
        ttk.Label(main_frame, text="Cargar pesos Preentrenado:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.directory_entry = ttk.Entry(main_frame, textvariable=self.model_path)
        self.directory_entry.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Seleccionar Directorio", command=self.abrir_archivo, style="TButton").grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)

        # Section Project Name wandb
        ttk.Label(main_frame, text="Nombre de proyecto wandb:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.guardar_entry = ttk.Entry(main_frame, textvariable=self.wandb_project_name)
        self.guardar_entry.grid(row=2, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Guardar", command=self.browse_guardar_directory, style="TButton").grid(row=2, column=2, padx=10, pady=10, sticky=tk.W)

        # Main progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(main_frame, text=f"Entrenando con un dispoditivo {self.device}").grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)

        # Progress and main task labels
        ttk.Label(main_frame, text="Progreso:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        ttk.Label(main_frame, textvariable=self.porcentaje).grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
        ttk.Label(main_frame, textvariable=self.tarea).grid(row=4, column=2, padx=10, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Valid list:").grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
        transpose_mask_checkbutton = ttk.Checkbutton(main_frame, variable=self.transpose_mask)
        transpose_mask_checkbutton.grid(row=5, column=2, padx=10, pady=5, sticky=(tk.W, tk.E))


        # Processing buttons
        ttk.Button(main_frame, text="Procesar Imágenes a 64 x 64 e entrenar", command=self.train_autoencoder_async, style="TButton").grid(row=8, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        # edit values button
        ttk.Button(main_frame, text="Editar Valores", command=self.editar_valores, style="TButton").grid(row=10, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Editar Valores train aug list", command=self.editar_valores_train_aug_list, style="TButton").grid(row=11, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

    def editar_valores(self):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Editar Valores")

        ttk.Label(edit_window, text="Learning Rate:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        learning_rate_entry = ttk.Entry(edit_window, textvariable=self.learning_rate)
        learning_rate_entry.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="Batch Size:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        batch_size_entry = ttk.Entry(edit_window, textvariable=self.batch_size)
        batch_size_entry.grid(row=1, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="Numero de carpetas por pasadas:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        num_carpetas_entry = ttk.Entry(edit_window, textvariable=self.num_carpetas_img)
        num_carpetas_entry.grid(row=2, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="size image:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        cluster_entry = ttk.Entry(edit_window, textvariable=self.size)
        cluster_entry.grid(row=3, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="epocas:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        cluster_entry = ttk.Entry(edit_window, textvariable=self.num_epochs)
        cluster_entry.grid(row=4, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(edit_window, text="Guardar", command=self.guardar_valores_en_archivo, style="TButton").grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(edit_window, text="cerrar", command=edit_window.destroy, style="TButton").grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

    def editar_valores_train_aug_list(self):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Editar Valores train aug list")

        ttk.Label(edit_window, text="mean:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        learning_rate_entry = ttk.Entry(edit_window, textvariable=self.mean)
        learning_rate_entry.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="std:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        batch_size_entry = ttk.Entry(edit_window, textvariable=self.std)
        batch_size_entry.grid(row=1, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="rotate limit:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        num_carpetas_entry = ttk.Entry(edit_window, textvariable=self.rotate_limit)
        num_carpetas_entry.grid(row=2, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="shift limit:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        cluster_entry = ttk.Entry(edit_window, textvariable=self.shift_limit)
        cluster_entry.grid(row=3, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="scale limit:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        cluster_entry = ttk.Entry(edit_window, textvariable=self.scale_limit)
        cluster_entry.grid(row=4, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="var limit:").grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
        var_limit_frame = ttk.Frame(edit_window)
        var_limit_frame.grid(row=5, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(var_limit_frame, text="Min:").grid(row=0, column=0, sticky=tk.W)
        var_limit_min_entry = ttk.Entry(var_limit_frame, textvariable=self.var_limit_min)
        var_limit_min_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        ttk.Label(var_limit_frame, text="Max:").grid(row=0, column=2, padx=5, sticky=tk.W)
        var_limit_max_entry = ttk.Entry(var_limit_frame, textvariable=self.var_limit_max)
        var_limit_max_entry.grid(row=0, column=3, padx=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="mask fill value:").grid(row=7, column=0, padx=10, pady=5, sticky=tk.W)
        cluster_entry = ttk.Entry(edit_window, textvariable=self.mask_fill_value)
        cluster_entry.grid(row=7, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(edit_window, text="transpose mask:").grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)
        transpose_mask_checkbutton = ttk.Checkbutton(edit_window, variable=self.transpose_mask)
        transpose_mask_checkbutton.grid(row=8, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))


        ttk.Button(edit_window, text="Guardar", command=self.guardar_valores_en_archivo, style="TButton").grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(edit_window, text="cerrar", command=edit_window.destroy, style="TButton").grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))


    def cargar_valores_desde_archivo(self):
        #(Load values from the configuration file)
        try:
            with open(f"{self.path}/config/configuracion_ajustes_entrenamiento.json", "r") as file:
                datos = json.load(file)
                self.directory_path.set(datos.get("directorio", ""))
                self.learning_rate.set(float(datos.get("learning_rate", 2e-5)))
                self.num_epochs.set(int(datos.get("num_epochs", 4)))
                self.batch_size.set(int(datos.get("batch_size", 64)))
                self.num_carpetas_img.set(int(datos.get("num_carpetas_img", 5)))
                self.cluster.set(int(datos.get("cluster", 2)))
                self.wandb_project_name.set(datos.get("wandb", "autoencoder_training"))
                self.umbral.set(int(datos.get("umbral de informacion", 10)))
                self.model_path.set(datos.get("model_path", ""))
                self.size.set(datos.get("size", 64))

                # Nuevos valores
                self.mean.set(datos.get("mean", 30))
                self.std.set(datos.get("std", 30))
                self.rotate_limit.set(datos.get("rotate_limit", 360))
                self.shift_limit.set(datos.get("shift_limit", 0.15))
                self.scale_limit.set(datos.get("scale_limit", 0.15))
                self.var_limit_min.set(datos.get("var_limit_min", 10))
                self.var_limit_max.set(datos.get("var_limit_max", 50))
                self.mask_fill_value.set(datos.get("mask_fill_value", 0))
                self.transpose_mask.set(datos.get("transpose_mask", True))

        except FileNotFoundError:
            self.guardar_valores_en_archivo()

    def guardar_valores_en_archivo(self):
        #(Save values to the configuration file)
        datos = {
            "directorio": self.directory_path.get(),
            "learning_rate": str(self.learning_rate.get()),
            "num_epochs": str(self.num_epochs.get()),
            "batch_size": str(self.batch_size.get()),
            "num_carpetas_img": str(self.num_carpetas_img.get()),
            "cluster": str(self.cluster.get()),
            "wandb": str(self.wandb_project_name.get()),
            "umbral de informacion": str(self.umbral.get()),
            "model_path": self.model_path.get(),
            "size": self.size.get(),
            "mean": int(self.mean.get()),
            "std": int(self.std.get()),
            "rotate_limit": self.rotate_limit.get(),
            "shift_limit": self.shift_limit.get(),
            "scale_limit": self.scale_limit.get(),
            "var_limit_min": int(self.var_limit_min.get()),
            "var_limit_max": int(self.var_limit_max.get()),
            "mask_fill_value": self.mask_fill_value.get(),
            "transpose_mask": self.transpose_mask.get(),
        }
        with open(f"{self.path}/config/configuracion_ajustes_entrenamiento.json", "w") as file:
            json.dump(datos, file)

    def browse_directory(self):
        #(Browse directory and update values)
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.directory_path.set(directory_path)
            self.guardar_valores_en_archivo()  # Guardar los valores actualizados en el archivo

    def abrir_archivo(self):
        #(Open file and update values)
        archivo_path = filedialog.askopenfilename()
        if archivo_path:
            self.model_path.set(archivo_path)
            self.guardar_valores_en_archivo()

    def browse_guardar_directory(self):
        #(Browse save directory and update values)
        self.wandb_project_name.set(self.guardar_entry.get())
        self.guardar_valores_en_archivo()  # Guardar los valores actualizados en el archivo

    def train_autoencoder_async(self):
        #(Start the training process asynchronously)
        threading.Thread(target=self.dividir_imagenes).start()  # Iniciar el proceso en un hilo diferente


    def dividir_imagenes(self):
        #(Logic for splitting and processing images)
        """
            Logic for splitting and processing images.

            This method retrieves the directory path, loads a pre-trained model, and processes
            subfolders within the specified path. For each subfolder, it updates progress variables,
            loads an image, and triggers the training of an autoencoder model.

            Returns:
                List: An empty list if an error occurs, otherwise, nothing.
        """
        ruta = self.directory_path.get()  # Get directory path from variable

        # Load a pre-trained model if the path is provided
        if self.model_path.get() != "":
            self.model = RegressionPLModel.load_from_checkpoint(self.model_path.get(),strict=False)
        
        # Check if a valid directory path is provided
        if ruta != "":
            try:
                # Get the list of folders at the specified path
                subcarpetas = [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]
                total_carpetas = len(subcarpetas)
                iter = 0

                # Iterate over subfolders
                for id_Ca, carpeta in enumerate(subcarpetas, start=1):
                    iter += 1
                    progreso_actual = (id_Ca / total_carpetas) * 100
                    # Update progress variables
                    self.porcentaje.set(f"{progreso_actual:.2f}%")
                    self.progress_var.set(progreso_actual)
                    self.tarea.set(f"Carpeta {id_Ca}/{total_carpetas}")

                    # Process every 'num_carpetas_img' subfolders
                    if iter == self.num_carpetas_img.get():
                        contenido = os.path.join(ruta, carpeta)
                        valid_mask_gt = cv2.imread(f"{contenido}/{carpeta}.png", 0)
                        pred_shape=valid_mask_gt.shape

                        # Create and configure a new autoencoder model
                        self.model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=64, learning_rate=self.learning_rate.get())
                        self.model.to(self.device)
                        # Trigger the training of the autoencoder
                        self.train_autoencoder(contenido,carpeta,valid_mask_gt)
                        iter = 0
                        
            except OSError as e:
                print(f"Error obtaining subfolders from {ruta}: {e}")
                return []
        else:
            print("No valid path specified.")
            return []
        
    def get_transforms(self,data):
        #(Get transformations based on the data type)
        if data == 'train':
            aug = A.Compose(self.train_aug_list)
        elif data == 'valid':
            aug = A.Compose(self.valid_aug_list)
        return aug 
        
    def read_image_mask(self,fragment_id,carpeta,start_idx=15,end_idx=45):
        #(Read images and masks from specified paths)

        """
            Read images and masks from specified paths.

            This method reads images and masks from the provided paths, processes them, and returns the results.

            Parameters:
                fragment_id (str): ID of the fragment.
                carpeta (str): Name of the folder.
                start_idx (int): Starting index for image loading.
                end_idx (int): Ending index for image loading.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: Images, mask, and fragment mask.
        """

        images = []

        # idxs = range(65)
        mid = 65 // 2
        start = mid - 30 // 2
        end = mid + 30 // 2
        idxs = range(start_idx, end_idx)

        for i in idxs:
            # Read and process each image
            image = cv2.imread(f"{fragment_id}/{i:02}.tif", 0)
            pad0 = (64- image.shape[0] % 64)
            pad1 = (64- image.shape[1] % 64)
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            image=np.clip(image,0,200)
            images.append(image)

        # Stack the images along the third axis
        images = np.stack(images, axis=2)

        # Read mask and fragment mask
        mask = cv2.imread(f"{fragment_id}/{carpeta}_inklabels.png", 0)

        fragment_mask=cv2.imread(f"{fragment_id}/{carpeta}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

        # Resize fragment mask and mask for specific cases
        kernel = np.ones((16,16),np.uint8)
        if 'frag' in fragment_id:
            fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

        # Normalize the mask
        mask = mask.astype('float32')
        mask/=255
        return images, mask,fragment_mask

    def get_train_valid_dataset(self,fragment_id,carpeta,mask,valid):
        #(Prepare training and validation datasets)

        """
            Prepare training and validation datasets.

            This method processes images and masks, generating training and validation datasets.

            Parameters:
                fragment_id (str): ID of the fragment.
                carpeta (str): Name of the folder.
                mask (np.ndarray): Mask array.
                valid (bool): Whether the dataset is for validation.

            Returns:
                Tuple[list, list, list, list, list]: Train images, train masks, valid images,
                valid masks, and valid xyxys.
        """
        train_images = []
        train_masks = []

        valid_images = []
        valid_masks = []
        valid_xyxys = []
        print('reading ',fragment_id)
        image, mask,fragment_mask= self.read_image_mask(fragment_id,carpeta)
        mask = mask.astype('float32')
        mask/=255
        x1_list = list(range(0, image.shape[1]-256+1, 32))
        y1_list = list(range(0, image.shape[0]-256+1, 32))

        # Iterate over positions to extract image patches
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,256,64):
                    for xi in range(0,256,64):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+self.size.get()
                        x2=x1+self.size.get()

                        # Check if the patch meets criteria for inclusion
                        if not np.all(mask[a:a + 256, b:b + 256]<0.05):
                                if not np.any(mask[a:a+ 256, b:b + 256]==0):
                                    train_images.append(image[y1:y2, x1:x2])
                                    train_masks.append(mask[y1:y2, x1:x2, None])
                                    assert image[y1:y2, x1:x2].shape==(self.size.get(),self.size.get(),30)
                        # If the dataset is for validation, also check fragment mask
                        if valid:
                            if not np.any(fragment_mask[a:a + 256, b:b + 256]==0):
                                    valid_images.append(image[y1:y2, x1:x2])
                                    valid_masks.append(mask[y1:y2, x1:x2, None])

                                    valid_xyxys.append([x1, y1, x2, y2])
                                    assert image[y1:y2, x1:x2].shape==(self.size,self.size,30)

        return train_images, train_masks, valid_images, valid_masks, valid_xyxys 
    
    def gkern(self,kernlen=21, nsig=3):
        #(Generate 2D Gaussian kernel)
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()

    def train_autoencoder(self,contenido,carpeta,mask):
        #(Training logic for the autoencoder)
        """
            Training logic for the autoencoder.

            This method prepares training and validation datasets, sets up data loaders,
            and trains the autoencoder model using PyTorch Lightning Trainer.

            Parameters:
                contenido (str): Path to the content.
                carpeta (str): Name of the folder.
                mask (np.ndarray): Mask array.

            Returns:
                None
        """
        # Prepare training and validation datasets
        train_images, train_masks, valid_images, valid_masks, valid_xyxys = self.get_train_valid_dataset(contenido,carpeta,mask,self.valid.get())

        # Create training dataset
        train_dataset = CustomDataset(
            train_images,labels=train_masks, transform=A.Compose(self.train_aug_list))
        
        # Create validation dataset if validation is enabled
        if self.valid.get():
            valid_dataset = CustomDataset(
                valid_images,xyxys=valid_xyxys, labels=valid_masks, transform = self.get_transforms(data='valid'))
            del valid_images, valid_masks, valid_xyxys
            gc.collect()

            # Create validation data loader
            valid_loader = DataLoader(valid_dataset,
                                batch_size=self.batch_size.get(),
                                shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=True,)

            del valid_dataset
            gc.collect()

        # Clean up memory
        del train_images, train_masks
        gc.collect()

        # Create training data loader
        train_loader = DataLoader(train_dataset,
                                    batch_size=self.batch_size.get(),
                                    shuffle=True,
                                    num_workers=4, pin_memory=True, drop_last=True,persistent_workers=True,
                                    )
        
        del train_dataset
        gc.collect()
        
        # Set up PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.num_epochs.get(),
            accelerator="gpu",
            devices=1,
            default_root_dir="./models",
            accumulate_grad_batches=1,
            precision='16-mixed',
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            callbacks=[ModelCheckpoint(filename=f'valid_'+'{epoch}',dirpath='./',monitor='train/Arcface_loss',mode='min',save_top_k=self.num_epochs.get()),],
        )

        # Train the model using PyTorch Lightning Trainer
        if self.valid.get():
            trainer.fit(model=self.model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            del valid_loader
            gc.collect()
        else:
            trainer.fit(model=self.model, train_dataloaders=train_loader)
            
        del train_loader
        gc.collect()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()