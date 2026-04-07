import torch
import torch.nn as nn
import torch.nn.functional as F

class CASSiPhysics(nn.Module):
    def __init__(self, mask_tensor, crf_np):
        super(CASSiPhysics, self).__init__()
        # Registramos la máscara y la curva como 'buffers'. 
        # Esto asegura que vayan automáticamente a la GPU sin ser entrenables.
        self.register_buffer('mask', mask_tensor) # [1, 1, H, W]
        self.register_buffer('crf', torch.from_numpy(crf_np)) # [3, 31]

    def shift_back(self, cassi_meas):
        """
        Paso 1: Convierte la imagen CASSI 2D [B, 1, H, W+30] 
        en el Cubo Ruidoso de 31 bandas [B, 31, H, W] para entrar a la red.
        """
        B, _, H, W_disp = cassi_meas.shape
        C = 31
        W = W_disp - C + 1
        
        cubo_inicial = torch.zeros((B, C, H, W), device=cassi_meas.device)
        for i in range(C):
            # Recortamos la ventana exacta para cada longitud de onda
            cubo_inicial[:, i, :, :] = cassi_meas[:, 0, :, i : i + W]
            
        return cubo_inicial

    def shift_forward(self, hsi_cube):
        """
        El Proceso Físico del Prisma: Dispersa el cubo 3D reconstruido [B, 31, H, W] 
        para crear una simulación CASSI 2D [B, 1, H, W+30].
        """
        B, C, H, W = hsi_cube.shape
        W_disp = W + C - 1
        
        # 1. Aplicar la máscara del sistema (Coded Aperture)
        masked_cube = hsi_cube * self.mask
        
        # 2. Dispersión (Prisma)
        cassi_sim = torch.zeros((B, 1, H, W_disp), device=hsi_cube.device)
        for i in range(C):
            cassi_sim[:, 0, :, i : i + W] += masked_cube[:, i, :, :]
            
        return cassi_sim

    def project_rgb(self, hsi_cube):
        """
        El Proceso Físico de la Cámara de Color: Aplasta el cubo 3D [B, 31, H, W]
        a un RGB [B, 3, H, W] usando la Curva de Eficiencia Cuántica (CRF).
        """
        B, C, H, W = hsi_cube.shape
        
        # Aplanamos el espacio para multiplicar píxel por píxel con la matriz de 3x31
        hsi_flat = hsi_cube.view(B, C, H * W) # [B, 31, H*W]
        
        # Multiplicación de matrices en PyTorch: (CRF) x (Cubo Aplanado)
        # crf.unsqueeze(0) permite que la misma curva se aplique a todo el batch
        rgb_sim_flat = torch.matmul(self.crf.unsqueeze(0), hsi_flat) # [B, 3, H*W]
        
        # Devolvemos la forma de imagen [B, 3, H, W]
        rgb_sim = rgb_sim_flat.view(B, 3, H, W)
        
        return rgb_sim

class SelfSupervisedLoss(nn.Module):
    def __init__(self, mask_tensor, crf_np):
        super(SelfSupervisedLoss, self).__init__()
        self.physics = CASSiPhysics(mask_tensor, crf_np)
        
    def forward(self, pred_cube, cassi_real, rgb_real):
        cassi_simulado = self.physics.shift_forward(pred_cube)
        loss_cassi = F.mse_loss(cassi_simulado, cassi_real)
        
        rgb_simulado = self.physics.project_rgb(pred_cube)
        loss_rgb = F.mse_loss(rgb_simulado, rgb_real)
        
        # ⚠️ EL CAMBIO: Le damos un megáfono de x10 o x50 al Juez de Color
        lambda_color = 20.0 
        loss_total = loss_cassi + (lambda_color * loss_rgb)
        
        return loss_total, loss_cassi, loss_rgb
        
        # Podríamos ponderarlas, pero Xie suele tratarlas igual.
        return loss_cassi + loss_rgb, loss_cassi, loss_rgb