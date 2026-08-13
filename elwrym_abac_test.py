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
        self.rgb_conv1 = nn.Conv2d(ch_rgb + 1, ch_rgb, kernel_size=3, padding=1, padding_mode='reflect')
        self.rgb_act = nn.LeakyReLU(0.2, inplace=True)
        self.rgb_conv2 = nn.Conv2d(ch_rgb, ch_rgb, kernel_size=3, padding=1, padding_mode='reflect')
        
        # --- MODULADORES OPTIMIZADOS (Corrección de la Sopa Borrosa) ---
        # Kernels 1x1 para ahorrar peso y no emborronar espacialmente
        self.gamma_conv = nn.Conv2d(ch_rgb, ch_hsi, kernel_size=1)
        self.beta_conv  = nn.Conv2d(ch_rgb, ch_hsi, kernel_size=1)
        
        # Inicialización en ceros: Empieza como conexión residual pura
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)
        
        # --- RAMA 1 (QUÍMICA): Procesamiento de CASSI ---
        self.hsi_depthwise = nn.Conv2d(ch_hsi, ch_hsi, kernel_size=3, padding=1, groups=ch_hsi, bias=False)
        self.hsi_bn = nn.BatchNorm2d(ch_hsi)
        self.hsi_pointwise = nn.Conv2d(ch_hsi, ch_hsi, kernel_size=1, bias=False)
        self.hsi_act = nn.ReLU(inplace=True)

        # =========================================================
        # ⚠️ NUEVO: EL PULIDOR ESPACIAL (Convolución Separable)
        # =========================================================
        self.rgb_separable = nn.Sequential(
            # 1. Depthwise: Afila bordes canal por canal (groups=16)
            nn.Conv2d(ch_rgb, ch_rgb, kernel_size=3, padding=1, groups=ch_rgb, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            # 2. Pointwise: Cruza la información para formar texturas complejas (SSIM)
            nn.Conv2d(ch_rgb, ch_rgb, kernel_size=1, bias=False)
        )

    def forward(self, x_hsi, x_rgb):
        # Guardamos residual solo para la rama química
        res_hsi = x_hsi
        
        # ---------------------------------------------------------
        # PUENTE 1 -> 2: El Grito de Auxilio (Incertidumbre)
        # ---------------------------------------------------------
        # Corrección de velocidad: torch.var evita la raíz cuadrada lenta
        incertidumbre = torch.var(x_hsi, dim=1, keepdim=True)
        
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
        
        # =========================================================
        # ⚠️ NUEVO: EXTRACCIÓN ESTRUCTURAL PARA EL SSIM
        # =========================================================
        # En lugar de enviar la suma cruda, la pasamos por el bloque separable
        # para que la red extraiga los nuevos micro-bordes.
        rgb_para_siguiente_bloque = self.rgb_separable(out_rgb + x_rgb)
        
        return out_hsi + res_hsi, rgb_para_siguiente_bloque

# =========================================================================
# 2. LA RED PRINCIPAL: ELWRYM-ABAC
# =========================================================================
class ELWRYM_ABAC(nn.Module):
    def __init__(self, num_bands=31, num_rgb_features=16, num_blocks=4):
        super(ELWRYM_ABAC, self).__init__()
        
        # Cabezas de Inicialización
        self.head_hsi = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, padding_mode='reflect')
        self.head_rgb = nn.Conv2d(3, num_rgb_features, kernel_size=3, padding=1, padding_mode='reflect')
        
        # Cuerpo Cooperativo
        self.blocks = nn.ModuleList([ABAC_Block(num_bands, num_rgb_features) for _ in range(num_blocks)])
        
        # Cola de Reconstrucción (Solo nos interesa el cubo de 31 bandas)
        self.tail_hsi = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, cassi_shiftback, rgb_real):
        x_hsi = self.head_hsi(cassi_shiftback)
        x_rgb = self.head_rgb(rgb_real)
        
        for block in self.blocks:
            x_hsi, x_rgb = block(x_hsi, x_rgb)
            
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
    print("🔬 INICIANDO DIAGNÓSTICO: ELWRYM-ABAC (EXPERIMENTAL SSIM)")
    print("="*55)
    
    modelo = ELWRYM_ABAC(num_bands=31, num_rgb_features=16, num_blocks=4)
    
    dummy_cassi = torch.randn(1, 31, 256, 256) 
    dummy_rgb = torch.randn(1, 3, 256, 256)    
    
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (total_params * 4) / (1024 ** 2)
    
    print(f"[Arquitectura] : ELWRYM-ABAC (Modificación Separable)")
    print("-" * 55)
    print(f"[Parámetros]   : {total_params:,}")
    print(f"[Peso en Disco]: {peso_mb:.4f} MB")
    
    if HAS_THOP:
        macs, params = profile(modelo, inputs=(dummy_cassi, dummy_rgb), verbose=False)
        gflops = (macs * 2) / (10**9) 
        print(f"[Complejidad]  : {gflops:.4f} GFLOPs")
    print("-" * 55)
    
    inicio = time.time()
    salida = modelo(dummy_cassi, dummy_rgb)
    fin = time.time()
    tiempo_ms = (fin - inicio) * 1000
    
    print(f"✅ ¡Flujo Exitoso! El bloque separable está operativo.")
    print(f"   Forma de Salida : {list(salida.shape)}")
    print(f"   Tiempo CPU (Fwd): {tiempo_ms:.2f} ms")
    print("="*55 + "\n")