import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importamos tu ecosistema ABAC
from dataset_dual import CASSIDualDataset
from elwrym_abac import ELWRYM_ABAC
from physics_loss import CASSiPhysics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎬 Iniciando Renderizado Pro ELWRYM-ABAC en: {device}")

    # =========================================================================
    # ⚙️ CONFIGURACIÓN DE RUTAS
    # =========================================================================
    RUTA_MODELO_PTH = r"checkpoints_elwrym\mejor_modelo.pth"
    RUTA_CARPETA_IMAGEN = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep\dataset\fortest\fake_and_real_food_ms"
    
    NOMBRE_GIF = "inferencia_completa.gif"
    DURACION_FRAME_MS = 200 # Un poco más lento para poder apreciar el mapa de error
    # =========================================================================

    if not os.path.exists(RUTA_MODELO_PTH):
        raise FileNotFoundError(f"No se encontró el modelo en: {RUTA_MODELO_PTH}")

    ruta_padre = os.path.dirname(RUTA_CARPETA_IMAGEN)
    nombre_imagen_objetivo = os.path.basename(RUTA_CARPETA_IMAGEN)

    print(f"📦 Cargando la imagen objetivo: {nombre_imagen_objetivo}...")
    dataset_test = CASSIDualDataset(root_dir=ruta_padre, patch_size=256, num_patches_per_img=1, is_train=False)

    idx_elegido = None
    for i, folder in enumerate(dataset_test.image_folders):
        if os.path.basename(folder) == nombre_imagen_objetivo:
            idx_elegido = i
            break
            
    if idx_elegido is None:
        raise ValueError(f"No se pudo encontrar '{nombre_imagen_objetivo}' en el dataset.")

    # 1. Preparar la Red
    modelo = ELWRYM_ABAC(num_bands=31, num_rgb_features=16, num_blocks=4).to(device)
    checkpoint = torch.load(RUTA_MODELO_PTH, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        modelo.load_state_dict(checkpoint['model_state_dict'])
    else:
        modelo.load_state_dict(checkpoint)
        
    modelo.eval()

    mascara_sistema = dataset_test.get_mask().to(device)
    crf_np = dataset_test.crf_np
    fisica = CASSiPhysics(mascara_sistema, crf_np).to(device)

    # 2. Inferencia Neuronal
    print("🧠 Calculando tensores 3D...")
    with torch.no_grad():
        cassi_real, rgb_real, gt_cube = dataset_test[idx_elegido]
        
        cassi_real = cassi_real.unsqueeze(0).to(device)
        rgb_real = rgb_real.unsqueeze(0).to(device)
        
        with torch.amp.autocast('cuda'):
            cassi_sb = fisica.shift_back(cassi_real)
            pred_3d = modelo(cassi_sb, rgb_real)

    cubo_gt = torch.clamp(gt_cube, 0, 1).cpu().numpy()
    cubo_pred = torch.clamp(pred_3d[0], 0, 1).cpu().numpy()
    longitudes_onda = np.arange(400, 710, 10)
    
    # =========================================================================
    # 3. EL ESTUDIO DE RENDERIZADO (El truco de no borrar la figura)
    # =========================================================================
    print("🎞️ Renderizando frames idénticos al visualizador interactivo...")
    frames_gif = []
    
    # Creamos el lienzo base (3 Paneles)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(top=0.85) # Espacio para el título maestro
    
    # Inicializamos la primera banda (400nm) para dejar el lienzo configurado
    fig.suptitle("Barrido Espectral - Longitud de Onda: 400 nm", fontsize=16, fontweight='bold')
    
    titulo_gt = axes[0].set_title("Verdad Original (400nm)")
    img_gt_plot = axes[0].imshow(cubo_gt[0], cmap='gray', vmin=0, vmax=1)
    axes[0].axis('off')
    
    titulo_pred = axes[1].set_title("Predicción ABAC (400nm)")
    img_pred_plot = axes[1].imshow(cubo_pred[0], cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')
    
    axes[2].set_title("Mapa de Error Absoluto")
    error_inicial = np.abs(cubo_gt[0] - cubo_pred[0])
    img_error_plot = axes[2].imshow(error_inicial, cmap='rainbow', vmin=0, vmax=1.0)
    fig.colorbar(img_error_plot, ax=axes[2], fraction=0.046, pad=0.04) # Barra fija
    axes[2].axis('off')
    
    plt.tight_layout()

    # 4. El Bucle de Actualización (Como si hicieras scroll súper rápido)
    for i in range(31):
        wl = longitudes_onda[i]
        
        # Inyectamos los nuevos datos en las gráficas ya existentes
        img_gt_plot.set_data(cubo_gt[i])
        img_pred_plot.set_data(cubo_pred[i])
        img_error_plot.set_data(np.abs(cubo_gt[i] - cubo_pred[i]))
        
        # Actualizamos los textos
        fig.suptitle(f"Barrido Espectral - Longitud de Onda: {wl} nm", fontsize=16, fontweight='bold')
        titulo_gt.set_text(f"Verdad Original ({wl}nm)")
        titulo_pred.set_text(f"Predicción ABAC ({wl}nm)")
        
        # Forzamos a matplotlib a redibujar el canvas en memoria
        fig.canvas.draw()
        
        # Capturamos la foto
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        frames_gif.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    # 5. Exportar Final
    print("💾 Empaquetando el GIF final...")
    frames_gif[0].save(
        NOMBRE_GIF,
        save_all=True,
        append_images=frames_gif[1:],
        duration=DURACION_FRAME_MS,
        loop=0
    )
    
    print(f"✅ ¡Completado! Abre '{NOMBRE_GIF}' para ver tu obra de arte matemática.")

if __name__ == "__main__":
    main()