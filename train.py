import os
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_metric

# Importamos nuestros módulos auditados
from dataset_dual import CASSIDualDataset
from metricas import calcular_psnr, calcular_sam
from physics_loss import CASSiPhysics, SelfSupervisedLoss  # <-- El Motor Físico
from elwrym_abac import ELWRYM_ABAC                      # <-- La Nueva Arquitectura

# =========================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# =========================================================
# NOTA: Tu modelo pesa 0.3MB. Podrías subir el BATCH_SIZE a 32 o 64 sin 
# quemar la RTX 4070, pero lo dejaremos en 8x2 para ser 100% fieles a Xie.
BATCH_SIZE = 16              
ACCUMULATION_STEPS = 1      
MAX_EPOCHS = 1200     
INITIAL_LR = 0.001
FACTOR_REDUCCION = 0.2
EPOCAS_REDUCCION = 300
FRECUENCIA_DASHBOARD = 10 
FRECUENCIA_SAVE_50 = 50   

RUTA_TRAIN = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep\dataset\fortrain"
RUTA_TEST  = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep\dataset\fortest"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Hardware activado: {device} (Lote Virtual: {BATCH_SIZE}x{ACCUMULATION_STEPS} = 16)")

    # =========================================================
    # 1. PREPARACIÓN DE DATOS
    # =========================================================
    dataset_train = CASSIDualDataset(RUTA_TRAIN, patch_size=256, num_patches_per_img=10)
    dataset_val   = CASSIDualDataset(RUTA_TEST, patch_size=256, num_patches_per_img=1)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    loader_val   = DataLoader(dataset_val, batch_size=1, shuffle=False)

    # =========================================================
    # 2. INICIALIZACIÓN (CREAR LOS RECIPIENTES)
    # =========================================================
    # Red ABAC 
    modelo = ELWRYM_ABAC(num_bands=31, num_rgb_features=16, num_blocks=4).to(device)
    
    # ⚠️ ¡AQUÍ NACE EL OPTIMIZADOR! Tiene que estar antes del Checkpoint
    optimizer = torch.optim.Adam(modelo.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCAS_REDUCCION, gamma=FACTOR_REDUCCION)

    mascara_sistema = dataset_train.get_mask().to(device)
    crf_np = dataset_train.crf_np

    fisica = CASSiPhysics(mascara_sistema, crf_np).to(device)
    juez_supremo = SelfSupervisedLoss(mascara_sistema, crf_np).to(device)

    scaler = torch.amp.GradScaler('cuda')

    historial = {'loss_t': [], 'loss_c': [], 'loss_g': [], 'psnr': [], 'sam': [], 'ssim': []}
    mejor_psnr = 0.0
    start_epoch = 1

    os.makedirs("checkpoints_elwrym", exist_ok=True)

    # =========================================================
    # 2.5 RECUPERACIÓN (LLENAR LOS RECIPIENTES)
    # =========================================================
    ruta_recuperacion = "checkpoints_elwrym/ultimo_checkpoint.pth"
    
    if os.path.exists(ruta_recuperacion):
        print(f"📥 ¡Cápsula del tiempo detectada! Cargando desde: {ruta_recuperacion}")
        
        # weights_only=False apaga la advertencia roja kilométrica de PyTorch
        checkpoint = torch.load(ruta_recuperacion, weights_only=False)
        
        # Ahora sí, el modelo y el optimizador existen para recibir datos
        modelo.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        mejor_psnr = checkpoint['mejor_psnr']
        historial = checkpoint['historial']
        
        # IMPORTANTE: Sincronizar el scheduler con la época actual
        scheduler.last_epoch = start_epoch - 1
        
        print(f"✅ Restaurado exitosamente. Retomando en la época {start_epoch}.")
    else:
        print("🌱 Iniciando entrenamiento desde cero.")
    # =========================================================
    # 3. BUCLE DE ENTRENAMIENTO
    # =========================================================
    # Cambia el '1' por 'start_epoch'
    for epoca in range(start_epoch, MAX_EPOCHS + 1):
        tiempo_inicio_epoca = time.time() 
        
        modelo.train()
        loss_epoch = 0.0; loss_gb_ep = 0.0; loss_cb_ep = 0.0
        
        loop = tqdm(loader_train, desc=f"Época [{epoca}/{MAX_EPOCHS}]", leave=False)
        optimizer.zero_grad() 
        
        for i, (cassi_real, rgb_real, _) in enumerate(loop):
            cassi_real = cassi_real.to(device)
            rgb_real = rgb_real.to(device)
            
            # ⚡ FASE FORWARD EN FP16 ⚡
            with torch.amp.autocast('cuda'):
                # A) Pre-procesamiento físico: Alinear el CASSI
                cassi_shiftback = fisica.shift_back(cassi_real)
                
                # B) Inferencia Dual Cooperativa
                prediccion_3d = modelo(cassi_shiftback, rgb_real)
                
                # C) El Castigo: Físico + Color
                loss_total, loss_gb, loss_cb = juez_supremo(prediccion_3d, cassi_real, rgb_real)
                
                loss_total_acc = loss_total / ACCUMULATION_STEPS
                
            # ⚡ FASE BACKWARD ESCALADA ⚡
            scaler.scale(loss_total_acc).backward()
            
            # Actualizamos pesos
            if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(loader_train)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
            
            loss_epoch += loss_total.item()
            loss_gb_ep += loss_gb.item()
            loss_cb_ep += loss_cb.item()
            
            loop.set_postfix(LossT=f"{(loss_epoch/(i+1)):.4f}")
        
        scheduler.step()
        lr_actual = optimizer.param_groups[0]['lr']
        
        historial['loss_t'].append(loss_epoch / len(loader_train))
        historial['loss_c'].append(loss_cb_ep / len(loader_train)) # Color
        historial['loss_g'].append(loss_gb_ep / len(loader_train)) # Grayscale/CASSI
        
        # =========================================================
        # 4. VALIDACIÓN Y MÉTRICAS
        # =========================================================
        modelo.eval()
        psnr_total = 0.0; sam_total = 0.0; ssim_total = 0.0
        vis_cassi = None; vis_rgb_r = None; vis_gt = None; vis_pred = None
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for cassi_val, rgb_val, gt_val in loader_val:
                    cassi_val = cassi_val.to(device)
                    rgb_val = rgb_val.to(device)
                    gt_val = gt_val.to(device)
                    
                    # Para la inferencia también necesitamos el ShiftBack
                    cassi_sb_val = fisica.shift_back(cassi_val)
                    pred_val = modelo(cassi_sb_val, rgb_val)
                    
                    psnr_total += calcular_psnr(pred_val, gt_val)
                    sam_total += calcular_sam(pred_val, gt_val)
                    
                    pred_np = torch.clamp(pred_val[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    gt_np = torch.clamp(gt_val[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    ssim_total += ssim_metric(gt_np, pred_np, data_range=1.0, channel_axis=2)
                    
                    vis_cassi = cassi_val
                    vis_rgb_r = rgb_val
                    vis_gt = gt_val
                    vis_pred = pred_val
                
        psnr_medio = psnr_total / len(loader_val)
        sam_medio = sam_total / len(loader_val)
        ssim_medio = ssim_total / len(loader_val)
        
        historial['psnr'].append(psnr_medio)
        historial['sam'].append(sam_medio)
        historial['ssim'].append(ssim_medio)
        
        # ⏱️ CÁLCULO DE TIEMPOS ⏱️
        tiempo_fin_epoca = time.time()
        duracion_epoca = tiempo_fin_epoca - tiempo_inicio_epoca
        tiempo_restante_segundos = duracion_epoca * (MAX_EPOCHS - epoca)
        formato_duracion = str(datetime.timedelta(seconds=int(duracion_epoca)))
        formato_restante = str(datetime.timedelta(seconds=int(tiempo_restante_segundos)))

        print(f"\nResumen Época {epoca} | LR: {lr_actual:.6f} | Loss: {historial['loss_g'][-1]:.4f} (CASSI) + {historial['loss_c'][-1]:.4f} (RGB)")
        print(f"Métricas (VS GT) | PSNR: {psnr_medio:.2f}dB | SSIM: {ssim_medio:.4f} | SAM: {sam_medio:.2f}°")
        print(f"⏱️ Tiempo: {formato_duracion} por época | Faltan aprox: {formato_restante}")
        
        # Guardamos la cápsula del tiempo (Checkpoint Completo) cada época
        checkpoint = {
            'epoch': epoca,
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mejor_psnr': mejor_psnr,
            'historial': historial
        }
        
        # Siempre guardamos el último para poder reanudar
        torch.save(checkpoint, "checkpoints_elwrym/ultimo_checkpoint.pth")
        
        if psnr_medio > mejor_psnr:
            mejor_psnr = psnr_medio
            # Guardamos una copia segura del mejor
            torch.save(checkpoint, "checkpoints_elwrym/mejor_modelo.pth")
            print("⭐ ¡Nuevo récord! Modelo ELWRYM guardado.")
            
        if epoca % FRECUENCIA_SAVE_50 == 0:
            torch.save(modelo.state_dict(), f"checkpoints_elwrym/modelo_ep{epoca}.pth")
            
        # =========================================================
        # 5. GENERACIÓN DEL DASHBOARD VISUAL
        # =========================================================
        if epoca % FRECUENCIA_DASHBOARD == 0:
            print("📸 Exportando Dashboard Cooperativo...")
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f"ELWRYM-ABAC (Dual-Branch + FP16) | Época: {epoca}", fontsize=18, fontweight='bold')

            axs[0, 0].imshow(vis_cassi[0, 0].cpu().numpy(), cmap='gray')
            axs[0, 0].set_title("Medición CASSI PAN")
            axs[0, 0].axis('off')

            rgb_r_plot = np.clip(vis_rgb_r[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
            axs[0, 1].imshow(rgb_r_plot)
            axs[0, 1].set_title("RGB Real (Ground Truth)")
            axs[0, 1].axis('off')

            # Usamos el motor físico para simular el RGB desde la predicción
            rgb_p_tensor = fisica.project_rgb(vis_pred.float())
            rgb_p_plot = np.clip(rgb_p_tensor[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
            axs[0, 2].imshow(rgb_p_plot)
            axs[0, 2].set_title("RGB Simulado (Predicción)")
            axs[0, 2].axis('off')

            b15_real = vis_gt[0, 15].cpu().numpy()
            b15_pred = vis_pred[0, 15].cpu().numpy()
            b15_comparacion = np.concatenate((b15_real, b15_pred), axis=1) 
            axs[0, 3].imshow(b15_comparacion, cmap='gray')
            axs[0, 3].set_title("Banda 15 - Real vs Pred (Búsqueda de Nitidez)")
            axs[0, 3].axis('off')

            axs[1, 0].plot(historial['loss_t'], 'k-', label='Total')
            axs[1, 0].plot(historial['loss_c'], 'r--', label='Loss Color')
            axs[1, 0].plot(historial['loss_g'], 'b--', label='Loss CASSI')
            axs[1, 0].set_title("Evolución de Pérdidas")
            axs[1, 0].grid(True); axs[1, 0].legend()

            axs[1, 1].plot(historial['psnr'], 'b-')
            axs[1, 1].set_title("PSNR (Mayor es Mejor)")
            axs[1, 1].grid(True)

            axs[1, 2].plot(historial['ssim'], 'g-')
            axs[1, 2].set_title("SSIM (Nitidez Estructural)")
            axs[1, 2].grid(True)

            axs[1, 3].plot(historial['sam'], 'purple')
            axs[1, 3].set_title("SAM (Fidelidad Química - Menor es Mejor)")
            axs[1, 3].grid(True)

            plt.tight_layout()
            plt.savefig(f"checkpoints_elwrym/dashboard_ep{epoca}.png", bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()