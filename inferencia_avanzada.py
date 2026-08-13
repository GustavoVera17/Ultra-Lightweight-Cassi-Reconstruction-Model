import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image
import matplotlib.patches as patches

# Importamos nuestros módulos del ecosistema ABAC
from elwrym_abac import ELWRYM_ABAC
from physics_loss_old1 import CASSiPhysics
from metricas import calcular_psnr, calcular_sam

# =========================================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# =========================================================================
RUTA_MODELO_PTH = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Prueba1_CASSIREC\Resultados_UltraLightweight\pesos\mejor_modelo.pth"
RUTA_TEST_DATASET = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep\dataset\fortest"

CRF_NP = np.array([
    [0.0073, 0.0326, 0.1146, 0.2238, 0.2319, 0.1408, 0.0545, 0.0063, 0.0016, 0.0016,
     0.0020, 0.0049, 0.0163, 0.0458, 0.1065, 0.2088, 0.3516, 0.5284, 0.7042, 0.8359,
     0.8876, 0.8407, 0.7186, 0.5513, 0.3840, 0.2458, 0.1472, 0.0844, 0.0470, 0.0253, 0.0135],
    [0.0001, 0.0004, 0.0016, 0.0039, 0.0069, 0.0125, 0.0232, 0.0438, 0.0841, 0.1417,
     0.2114, 0.3160, 0.4578, 0.6127, 0.7303, 0.7711, 0.7259, 0.6094, 0.4566, 0.2974,
     0.1691, 0.0863, 0.0410, 0.0188, 0.0084, 0.0037, 0.0016, 0.0007, 0.0003, 0.0001, 0.0000],
    [0.0336, 0.1558, 0.5847, 1.2588, 1.4883, 1.1396, 0.6908, 0.3546, 0.1584, 0.0658,
     0.0267, 0.0099, 0.0038, 0.0015, 0.0005, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
], dtype=np.float32)
CRF_NP = CRF_NP / np.max(np.sum(CRF_NP, axis=1))

def calcular_ssim(pred_np, target_np):
    if pred_np.ndim == 3:
        ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    else:
        ssim_val = ssim_metric(target_np, pred_np, data_range=1.0)
    return float(ssim_val)

def calcular_psnr_np(pred_np, target_np):
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calcular_mae_np(pred_np, target_np):
    return np.mean(np.abs(pred_np - target_np))

def cargar_imagen_completa(folder_path):
    search_pattern = os.path.join(folder_path, '**', '*.png')
    all_pngs = sorted(glob.glob(search_pattern, recursive=True))
    band_files = [f for f in all_pngs if os.path.splitext(os.path.basename(f))[0][-2:].isdigit()][:31]
    
    cube = []
    for file in band_files:
        img_array = np.array(Image.open(file), dtype=np.float32)
        if img_array.ndim == 3: img_array = img_array[:, :, 0]
        max_val = 65535.0 if np.max(img_array) > 255.0 else 255.0
        cube.append(img_array / max_val)
    return np.stack(cube, axis=2)

def simular_cassi_rgb(hsi_cube):
    H, W, C = hsi_cube.shape
    np.random.seed(42)
    mask_np = np.random.binomial(1, 0.5, (H, W)).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    
    hsi_flat = hsi_cube.reshape(-1, C).T 
    rgb_flat = np.dot(CRF_NP, hsi_flat) 
    rgb_patch = rgb_flat.T.reshape(H, W, 3)
    
    masked_patch = hsi_cube * mask_np[:, :, np.newaxis]
    cassi_meas = np.zeros((H, W + C - 1), dtype=np.float32)
    for i in range(C):
        cassi_meas[:, i:(i + W)] += masked_patch[:, :, i]
        
    return torch.from_numpy(cassi_meas).unsqueeze(0).unsqueeze(0), \
           torch.from_numpy(rgb_patch).permute(2, 0, 1).unsqueeze(0), \
           torch.from_numpy(hsi_cube).permute(2, 0, 1).unsqueeze(0), \
           mask_tensor

# =========================================================================
# 📊 INTERFACES MATPLOTLIB
# =========================================================================

class MetricsPlotter:
    def __init__(self, psnrs, ssims, maes, img_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.canvas.manager.set_window_title(f"Métricas por Banda - {img_name}")
        fig.suptitle(f"Análisis de Rendimiento por Banda - {img_name}", fontsize=16)
        
        bandas = np.arange(400, 710, 10)
        
        axes[0].plot(bandas, psnrs, 'b-o', linewidth=2)
        axes[0].set_title("PSNR por Banda (Más alto es mejor)")
        axes[0].set_xlabel("Longitud de Onda (nm)")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].grid(True, alpha=0.5)
        
        axes[1].plot(bandas, ssims, 'g-o', linewidth=2)
        axes[1].set_title("SSIM por Banda (Más alto es mejor)")
        axes[1].set_xlabel("Longitud de Onda (nm)")
        axes[1].set_ylabel("SSIM")
        axes[1].grid(True, alpha=0.5)
        
        axes[2].plot(bandas, maes, 'r-o', linewidth=2)
        axes[2].set_title("MAE por Banda (Más bajo es mejor)")
        axes[2].set_xlabel("Longitud de Onda (nm)")
        axes[2].set_ylabel("MAE Absoluto")
        axes[2].grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.show()

class SpectralInspector:
    def __init__(self, rgb_img, gt_cube, pred_cube, img_name):
        self.rgb_img = np.clip(rgb_img, 0, 1)
        self.gt_cube = gt_cube
        self.pred_cube = pred_cube
        
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(f"Firmas Espectrales - {img_name}")
        self.fig.suptitle("Inspector de Firmas Espectrales (Haz clic en RGB para analizar)", fontsize=16)
        
        # Disposición similar a la imagen objetivo
        self.ax_img = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1)
        self.ax_img.imshow(self.rgb_img)
        self.ax_img.set_title("Imagen RGB Simulada")
        self.ax_img.axis('off')
        
        self.ax_plots = [
            plt.subplot2grid((2, 3), (0, 0)),
            plt.subplot2grid((2, 3), (0, 2)),
            plt.subplot2grid((2, 3), (1, 0)),
            plt.subplot2grid((2, 3), (1, 1)),
            plt.subplot2grid((2, 3), (1, 2))
        ]
        
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.points = [(None, None)] * 5
        self.markers = [None] * 5
        self.click_count = 0
        
        for ax in self.ax_plots:
            ax.set_ylim(0, 1)
            ax.set_xlim(400, 700)
            ax.set_xlabel("Longitud de Onda (nm)")
            ax.set_ylabel("Reflectancia")
            ax.set_visible(False)
            
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.tight_layout()
        plt.show()
        
    def onclick(self, event):
        if event.inaxes != self.ax_img: return
        x, y = int(event.xdata), int(event.ydata)
        
        idx = self.click_count % 5
        color = self.colors[idx]
        
        self.points[idx] = (x, y)
        
        if self.markers[idx] is not None:
            self.markers[idx].remove()
            
        self.markers[idx] = self.ax_img.plot(x, y, 's', color=color, markersize=8, markeredgecolor='white')[0]
        self.click_count += 1
        
        self.update_plots()
        self.fig.canvas.draw_idle()
        
    def update_plots(self):
        longitudes = np.arange(400, 710, 10)
        for i, ax in enumerate(self.ax_plots):
            if self.points[i][0] is not None:
                x, y = self.points[i]
                c = self.colors[i]
                gt_sig = self.gt_cube[:, y, x]
                pred_sig = self.pred_cube[:, y, x]
                
                ax.clear()
                ax.plot(longitudes, gt_sig, label="Reference (GT)", linewidth=2, color='#1f77b4')
                ax.plot(longitudes, pred_sig, label="Ours (ABAC)", linewidth=2, color='#ff7f0e')
                ax.set_title(f"Punto {i+1} [x:{x}, y:{y}]", color=c)
                ax.set_ylim(0, 1)
                ax.set_xlim(400, 700)
                ax.set_xlabel("Longitud de Onda (nm)")
                ax.set_ylabel("Reflectancia")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_visible(True)


class SpatialExplorer:
    def __init__(self, gt_cube, pred_cube, img_name):
        self.gt_cube = gt_cube
        self.pred_cube = pred_cube
        self.banda_actual = 15
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.canvas.manager.set_window_title(f"Explorador Espacial - {img_name}")
        plt.subplots_adjust(bottom=0.2)
        
        self.ax_gt = self.axes[0, 0]
        self.ax_pred = self.axes[0, 1]
        self.ax_err = self.axes[0, 2]
        self.ax_zoom_gt = self.axes[1, 0]
        self.ax_zoom_pred = self.axes[1, 1]
        self.ax_zoom_err = self.axes[1, 2]
        
        self.img_gt = self.ax_gt.imshow(self.gt_cube[self.banda_actual], cmap='gray', vmin=0, vmax=1)
        self.img_pred = self.ax_pred.imshow(self.pred_cube[self.banda_actual], cmap='gray', vmin=0, vmax=1)
        
        # ERROR MAP CON RAINBOW Y VMAX=1.0 COMO SOLICITADO
        err = np.abs(self.gt_cube[self.banda_actual] - self.pred_cube[self.banda_actual])
        self.img_err = self.ax_err.imshow(err, cmap='rainbow', vmin=0, vmax=1.0)
        self.fig.colorbar(self.img_err, ax=self.ax_err, fraction=0.046, pad=0.04)
        
        self.ax_gt.set_title("Ground Truth (Selecciona Zoom aquí)")
        self.ax_pred.set_title("ABAC Prediction")
        self.ax_err.set_title("Error Absoluto (max 1.0)")
        
        for ax in [self.ax_gt, self.ax_pred, self.ax_err, self.ax_zoom_gt, self.ax_zoom_pred, self.ax_zoom_err]:
            ax.axis('off')
            
        # Zonas de zoom (inicializado en negro)
        self.zoom_gt = self.ax_zoom_gt.imshow(np.zeros((50,50)), cmap='gray', vmin=0, vmax=1)
        self.zoom_pred = self.ax_zoom_pred.imshow(np.zeros((50,50)), cmap='gray', vmin=0, vmax=1)
        self.zoom_err = self.ax_zoom_err.imshow(np.zeros((50,50)), cmap='rainbow', vmin=0, vmax=1.0)
        
        self.ax_zoom_gt.set_title("Zoom GT")
        self.ax_zoom_pred.set_title("Zoom Pred")
        self.ax_zoom_err.set_title("Zoom Error")
        
        self.rect_gt = None
        self.rect_pred = None
        
        self.rs = RectangleSelector(self.ax_gt, self.onselect, useblit=True,
                                    button=[1], minspanx=10, minspany=10,
                                    spancoords='pixels', interactive=True)
                                    
        axcolor = 'lightgoldenrodyellow'
        self.ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
        self.slider = Slider(self.ax_slider, 'Banda', 0, 30, valinit=15, valstep=1)
        self.slider.on_changed(self.update_band)
        
        # Inicializar con un recuadro de zoom por defecto en el centro
        H, W = self.gt_cube.shape[1], self.gt_cube.shape[2]
        self.current_extents = (W//2 - 50, W//2 + 50, H//2 - 50, H//2 + 50)
        self.update_zooms()
        self.draw_red_rectangles()
        
        plt.show()
        
    def draw_red_rectangles(self):
        if self.current_extents is None: return
        x1, x2, y1, y2 = self.current_extents
        
        if self.rect_pred is not None:
            self.rect_pred.remove()
            
        self.rect_pred = patches.Rectangle((min(x1,x2), min(y1,y2)), abs(x2-x1), abs(y2-y1), 
                                 linewidth=2, edgecolor='red', facecolor='none')
        self.ax_pred.add_patch(self.rect_pred)
        
    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.current_extents = (min(x1,x2), max(x1,x2), min(y1,y2), max(y1,y2))
        self.update_zooms()
        self.draw_red_rectangles()
        self.fig.canvas.draw_idle()
        
    def update_zooms(self):
        if self.current_extents is None: return
        x1, x2, y1, y2 = self.current_extents
        b = self.banda_actual
        
        z_gt = self.gt_cube[b, y1:y2, x1:x2]
        z_pred = self.pred_cube[b, y1:y2, x1:x2]
        z_err = np.abs(z_gt - z_pred)
        
        self.zoom_gt.set_data(z_gt)
        self.zoom_pred.set_data(z_pred)
        self.zoom_err.set_data(z_err)
        
        self.zoom_gt.set_extent((0, x2-x1, y2-y1, 0))
        self.zoom_pred.set_extent((0, x2-x1, y2-y1, 0))
        self.zoom_err.set_extent((0, x2-x1, y2-y1, 0))
        self.fig.canvas.draw_idle()

    def update_band(self, val):
        self.banda_actual = int(val)
        b = self.banda_actual
        self.img_gt.set_data(self.gt_cube[b])
        self.img_pred.set_data(self.pred_cube[b])
        self.img_err.set_data(np.abs(self.gt_cube[b] - self.pred_cube[b]))
        self.ax_gt.set_title(f"Ground Truth ({400 + b*10}nm)")
        self.ax_pred.set_title(f"ABAC Prediction ({400 + b*10}nm)")
        self.update_zooms()
        self.fig.canvas.draw_idle()

# =========================================================================
# 🚀 FUNCIÓN PRINCIPAL
# =========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print("🔬 INFERENCIA AVANZADA ELWRYM-ABAC")
    print("="*50)
    
    image_folders = [f.path for f in os.scandir(RUTA_TEST_DATASET) if f.is_dir()]
    if not image_folders:
        print(f"No se encontraron imágenes en {RUTA_TEST_DATASET}")
        return
        
    print("\nImágenes disponibles:")
    for i, folder in enumerate(image_folders):
        print(f"  [{i}] {os.path.basename(folder)}")
        
    seleccion = input("\nIntroduce el número de la imagen a evaluar (o 'q' para salir): ")
    if seleccion.lower() == 'q': return
    try:
        idx = int(seleccion)
        carpeta_elegida = image_folders[idx]
        nombre_img = os.path.basename(carpeta_elegida)
    except:
        print("Selección inválida.")
        return

    print(f"\nCargando cubo HSI completo de '{nombre_img}'...")
    hsi_cube_np = cargar_imagen_completa(carpeta_elegida)
    print(f"Dimensiones del cubo original: {hsi_cube_np.shape} (Full Size)")
    
    print("Simulando mediciones CASSI y RGB dinámicas...")
    cassi_tensor, rgb_tensor, gt_tensor, mask_tensor = simular_cassi_rgb(hsi_cube_np)
    
    cassi_tensor = cassi_tensor.to(device)
    rgb_tensor = rgb_tensor.to(device)
    gt_tensor = gt_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    print("Cargando modelo...")
    modelo = ELWRYM_ABAC(num_bands=31, num_rgb_features=16, num_blocks=4).to(device)
    checkpoint = torch.load(RUTA_MODELO_PTH, map_location=device, weights_only=False)
    modelo.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    modelo.eval()
    
    fisica = CASSiPhysics(mask_tensor, CRF_NP).to(device)
    
    print("Ejecutando red neuronal...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cassi_sb = fisica.shift_back(cassi_tensor)
            pred_tensor = modelo(cassi_sb, rgb_tensor)
            
    pred_np = torch.clamp(pred_tensor[0], 0, 1).cpu().numpy() 
    gt_np = gt_tensor[0].cpu().numpy() 
    rgb_img_np = rgb_tensor[0].cpu().numpy().transpose(1, 2, 0) 
    
    print("\n" + "="*40)
    print("📊 MÉTRICAS POR BANDA (PSNR | SSIM | MAE)")
    print("="*40)
    print(f"{'Banda':<8} | {'Long. Onda':<12} | {'PSNR':<8} | {'SSIM':<8} | {'MAE':<8}")
    print("-" * 50)
    
    psnrs, ssims, maes = [], [], []
    for b in range(31):
        p = calcular_psnr_np(pred_np[b], gt_np[b])
        s = calcular_ssim(pred_np[b], gt_np[b])
        m = calcular_mae_np(pred_np[b], gt_np[b])
        psnrs.append(p); ssims.append(s); maes.append(m)
        wl = 400 + b*10
        print(f"Band {b:02d} | {wl}nm       | {p:.2f}  | {s:.4f}   | {m:.4f}")
        
    print("-" * 50)
    print(f"GLOBAL   | 400-700nm    | {np.mean(psnrs):.2f}  | {np.mean(ssims):.4f}   | {np.mean(maes):.4f}")
    
    print("\nIniciando Interfaces Visuales (Cierra una ventana para pasar a la siguiente)...")
    
    print("1️⃣ Abriendo 'Gráficas de Métricas'...")
    MetricsPlotter(psnrs, ssims, maes, nombre_img)
    
    print("2️⃣ Abriendo 'Inspector de Firmas Espectrales'...")
    SpectralInspector(rgb_img_np, gt_np, pred_np, nombre_img)
    
    print("3️⃣ Abriendo 'Explorador Espacial'...")
    SpatialExplorer(gt_np, pred_np, nombre_img)
    print("¡Análisis Finalizado!")

if __name__ == "__main__":
    main()
