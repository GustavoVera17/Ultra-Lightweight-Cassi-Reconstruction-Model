import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. EL BLOQUE COOPERATIVO (Atención Bidireccional Asimétrica)
# =========================================================================
class ABAC_Block(nn.Module):
    def __init__(self, ch_hsi=31, ch_rgb=16):
        super(ABAC_Block, self).__init__()
        
        # --- RAMA 2 (ESPACIAL): Procesamiento de Geometría ---
        # Recibe 16 canales RGB + 1 canal de Incertidumbre de la Rama 1
        self.rgb_conv1 = nn.Conv2d(ch_rgb + 1, ch_rgb, kernel_size=3, padding=1, padding_mode='reflect')
        self.rgb_act = nn.LeakyReLU(0.2, inplace=True)
        self.rgb_conv2 = nn.Conv2d(ch_rgb, ch_rgb, kernel_size=3, padding=1, padding_mode='reflect')
        
        # Generadores de Modulación Espacial (De Rama 2 a Rama 1)
        self.gamma_conv = nn.Conv2d(ch_rgb, ch_hsi, kernel_size=3, padding=1)
        self.beta_conv  = nn.Conv2d(ch_rgb, ch_hsi, kernel_size=3, padding=1)
        
        # --- RAMA 1 (QUÍMICA): Procesamiento de CASSI ---
        # Convoluciones ultraligeras para limpieza de espectro
        self.hsi_depthwise = nn.Conv2d(ch_hsi, ch_hsi, kernel_size=3, padding=1, groups=ch_hsi, bias=False)
        self.hsi_bn = nn.BatchNorm2d(ch_hsi)
        self.hsi_pointwise = nn.Conv2d(ch_hsi, ch_hsi, kernel_size=1, bias=False)
        self.hsi_act = nn.ReLU(inplace=True)

    def forward(self, x_hsi, x_rgb):
        # Guardamos residuales para el flujo de gradientes
        res_hsi = x_hsi
        res_rgb = x_rgb
        
        # ---------------------------------------------------------
        # PUENTE 1 -> 2: El Grito de Auxilio (Incertidumbre)
        # ---------------------------------------------------------
        # Varianza espectral: zonas ruidosas brillan [Batch, 1, H, W]
        incertidumbre = torch.std(x_hsi, dim=1, keepdim=True)
        
        # ---------------------------------------------------------
        # RAMA 2: Entendiendo la Geometría
        # ---------------------------------------------------------
        rgb_fused = torch.cat([x_rgb, incertidumbre], dim=1)
        out_rgb = self.rgb_act(self.rgb_conv1(rgb_fused))
        out_rgb = self.rgb_conv2(out_rgb)
        
        # ---------------------------------------------------------
        # PUENTE 2 -> 1: La Guía Afín
        # ---------------------------------------------------------
        gamma = self.gamma_conv(out_rgb)
        beta  = self.beta_conv(out_rgb)
        
        # ---------------------------------------------------------
        # RAMA 1: Extracción Química Moldeada
        # ---------------------------------------------------------
        out_hsi = self.hsi_depthwise(x_hsi)
        out_hsi = self.hsi_bn(out_hsi)
        
        # ¡La Inyección Asimétrica!
        out_hsi = out_hsi * (1 + gamma) + beta
        
        out_hsi = self.hsi_act(out_hsi)
        out_hsi = self.hsi_pointwise(out_hsi)
        
        return out_hsi + res_hsi, out_rgb + res_rgb

# =========================================================================
# 2. LA RED PRINCIPAL: ELWRYM-ABAC
# =========================================================================
class ELWRYM_ABAC(nn.Module):
    def __init__(self, num_bands=31, num_rgb_features=16, num_blocks=4):
        super(ELWRYM_ABAC, self).__init__()
        
        # Cabezas de Inicialización
        self.head_hsi = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, padding_mode='reflect')
        # Expandimos los 3 canales crudos a los 16 del espacio latente geométrico
        self.head_rgb = nn.Conv2d(3, num_rgb_features, kernel_size=3, padding=1, padding_mode='reflect')
        
        # Cuerpo Cooperativo
        self.blocks = nn.ModuleList([ABAC_Block(num_bands, num_rgb_features) for _ in range(num_blocks)])
        
        # Cola de Reconstrucción (Solo nos interesa el cubo de 31 bandas)
        self.tail_hsi = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, cassi_shiftback, rgb_real):
        # 1. Inicialización en paralelo
        x_hsi = self.head_hsi(cassi_shiftback)
        x_rgb = self.head_rgb(rgb_real)
        
        # 2. Diálogo progresivo bloque por bloque
        for block in self.blocks:
            x_hsi, x_rgb = block(x_hsi, x_rgb)
            
        # 3. Salida Final
        out = self.tail_hsi(x_hsi)
        return out

# =========================================================================
# 3. ESCÁNER DE RENDIMIENTO Y PRUEBA DE TENSORES
# =========================================================================
if __name__ == "__main__":
    import time
    try:
        from thop import profile  # type: ignore
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False
        print("💡 Advertencia: Para medir FLOPs instala thop ('pip install thop')")

    print("\n" + "="*55)
    print("🔬 INICIANDO DIAGNÓSTICO: ELWRYM-ABAC (Dual-Branch)")
    print("="*55)
    
    # 1. Instanciamos el modelo con las Ramas Asimétricas
    modelo = ELWRYM_ABAC(num_bands=31, num_rgb_features=16, num_blocks=4)
    
    # 2. Generamos tensores simulados (Batch de 1 imagen, resolución 256x256)
    dummy_cassi = torch.randn(1, 31, 256, 256) # Rama Química
    dummy_rgb = torch.randn(1, 3, 256, 256)    # Rama Espacial (Cruda)
    
    # 3. Cálculo de Parámetros y Peso en Memoria (MB)
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (total_params * 4) / (1024 ** 2)
    
    print(f"[Arquitectura] : ELWRYM-ABAC (Cooperativa)")
    print(f"[Entrada HSI]  : CASSI ShiftBack {list(dummy_cassi.shape)}")
    print(f"[Entrada RGB]  : RGB Crudo       {list(dummy_rgb.shape)}")
    print("-" * 55)
    print(f"[Parámetros]   : {total_params:,}")
    print(f"[Peso en Disco]: {peso_mb:.4f} MB")
    
    # 4. Cálculo de FLOPs
    if HAS_THOP:
        macs, params = profile(modelo, inputs=(dummy_cassi, dummy_rgb), verbose=False)
        gflops = (macs * 2) / (10**9) 
        print(f"[Complejidad]  : {gflops:.4f} GFLOPs")
    print("-" * 55)
    
    # 5. Prueba de Flujo y Velocidad (Forward Pass)
    print("\nEjecutando prueba de diálogo de tensores...")
    inicio = time.time()
    
    salida = modelo(dummy_cassi, dummy_rgb)
    
    fin = time.time()
    tiempo_ms = (fin - inicio) * 1000
    
    print(f"✅ ¡Flujo Exitoso! Las ramas no colapsaron.")
    print(f"   Forma de Salida : {list(salida.shape)} (Debe ser [1, 31, 256, 256])")
    print(f"   Tiempo CPU (Fwd): {tiempo_ms:.2f} ms")
    print("="*55 + "\n")