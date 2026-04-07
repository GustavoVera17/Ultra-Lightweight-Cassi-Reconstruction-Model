import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. EL GENERADOR DE BORDES INTERNO (Sobel Fijo)
# =========================================================================
class SobelExtractor(nn.Module):
    """Extrae el mapa de bordes (1 canal) a partir del RGB (3 canales)"""
    def __init__(self):
        super(SobelExtractor, self).__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)

    def forward(self, rgb):
        # Convertimos RGB a escala de grises para un solo mapa de Sobel contundente
        gray = 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]
        grad_x = F.conv2d(gray, self.weight_x, padding=1)
        grad_y = F.conv2d(gray, self.weight_y, padding=1)
        # Magnitud del borde (1 solo canal de salida)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

# =========================================================================
# 2. EL BLOQUE DE INYECCIÓN PERPENDICULAR (Modulación)
# =========================================================================
class GuidedELWRYMBlock(nn.Module):
    def __init__(self, channels):
        super(GuidedELWRYMBlock, self).__init__()
        # --- Arteria Principal (Procesamiento Químico) ---
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # --- Vena Perpendicular (El Modulador de Sobel) ---
        # Entra 1 canal (Sobel), salen 'channels' canales para Gamma y Beta
        self.gamma_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)

    def forward(self, x, sobel_map):
        residual = x
        
        # 1. Extracción de características del CASSI
        out = self.relu(self.bn1(self.depthwise(x)))
        
        # 2. INYECCIÓN PERPENDICULAR (Modulación Espacial)
        gamma = self.gamma_conv(sobel_map)
        beta = self.beta_conv(sobel_map)
        out = out * (1 + gamma) + beta  
        
        # 3. Mezcla de canales y salida
        out = self.bn2(self.pointwise(out))
        out += residual
        return self.relu(out)

# =========================================================================
# 3. LA RED PRINCIPAL: ELWRYM-GUIDED
# =========================================================================
class ELWRYM_Guided(nn.Module):
    def __init__(self, num_bands=31, num_features=96, num_blocks=4):
        super(ELWRYM_Guided, self).__init__()
        self.sobel_extractor = SobelExtractor()
        
        # Entrada: SÓLO los 31 canales del CASSI ShiftBack
        self.head = nn.Sequential(
            nn.Conv2d(num_bands, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        
        # Lista de bloques guiados
        self.blocks = nn.ModuleList([GuidedELWRYMBlock(num_features) for _ in range(num_blocks)])
        
        self.tail = nn.Conv2d(num_features, num_bands, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, cassi_shiftback, rgb_real):
        # 1. Extraemos el mapa topográfico (Sobel) del RGB de la cámara
        sobel_map = self.sobel_extractor(rgb_real)
        
        # 2. El CASSI entra a la arteria principal
        x = self.head(cassi_shiftback)
        
        # 3. Procesamiento profundo con inyección perpendicular en CADA bloque
        for block in self.blocks:
            x = block(x, sobel_map)
            
        # 4. Reconstrucción final
        out = self.tail(x)
        return out

# =========================================================================
# 4. ESCÁNER DE RENDIMIENTO Y PRUEBA DE TENSORES
# =========================================================================
if __name__ == "__main__":
    import time
    try:
        from thop import profile # type: ignore
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False
        print("💡 Advertencia: Para medir FLOPs instala thop ('pip install thop')")

    print("\n" + "="*55)
    print("🔬 INICIANDO DIAGNÓSTICO: ELWRYM-GUIDED")
    print("="*55)
    
    # 1. Instanciamos el modelo con la configuración propuesta
    # (96 canales internos le dan la potencia necesaria para asimilar el Sobel)
    modelo = ELWRYM_Guided(num_bands=31, num_features=96, num_blocks=4)
    
    # 2. Generamos tensores simulados (Batch de 1 imagen, resolución 256x256)
    dummy_cassi = torch.randn(1, 31, 256, 256) # El puré del ShiftBack
    dummy_rgb = torch.randn(1, 3, 256, 256)    # La cámara RGB nítida
    
    # 3. Cálculo de Parámetros y Peso en Memoria (MB)
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (total_params * 4) / (1024 ** 2) # 4 bytes por parámetro (Float32)
    
    print(f"[Arquitectura] : ELWRYM-Guided (Spatially-Adaptive)")
    print(f"[Entrada 1]    : CASSI ShiftBack {list(dummy_cassi.shape)}")
    print(f"[Entrada 2]    : RGB Cámara      {list(dummy_rgb.shape)}")
    print("-" * 55)
    print(f"[Parámetros]   : {total_params:,}")
    print(f"[Peso en Disco]: {peso_mb:.4f} MB")
    
    # 4. Cálculo de FLOPs
    if HAS_THOP:
        macs, params = profile(modelo, inputs=(dummy_cassi, dummy_rgb), verbose=False)
        # 1 MAC (Multiply-Accumulate) = 2 FLOPs aproximadamente
        gflops = (macs * 2) / (10**9) 
        print(f"[Complejidad]  : {gflops:.4f} GFLOPs")
    print("-" * 55)
    
    # 5. Prueba de Flujo y Velocidad (Forward Pass)
    print("\nEjecutando prueba de flujo de tensores...")
    inicio = time.time()
    
    # Simulamos el paso de los tensores por la red
    salida = modelo(dummy_cassi, dummy_rgb)
    
    fin = time.time()
    tiempo_ms = (fin - inicio) * 1000
    
    print(f"✅ ¡Flujo Exitoso!")
    print(f"   Forma de Salida : {list(salida.shape)} (Debe ser [1, 31, 256, 256])")
    print(f"   Tiempo CPU (Fwd): {tiempo_ms:.2f} ms")
    print("="*55 + "\n")