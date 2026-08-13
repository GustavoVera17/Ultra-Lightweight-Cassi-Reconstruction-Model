import torch
from elwrym_abac_test import ELWRYM_ABAC

print("📸 Generando radiografía 3D de la arquitectura asimétrica ELWRYM-ABAC...")

# 1. Instanciamos tu modelo con los mismos parámetros
modelo = ELWRYM_ABAC(num_bands=31, num_rgb_features=16, num_blocks=4)
modelo.eval() # Siempre poner en modo evaluación antes de exportar

# =====================================================================
# ⚠️ EL CAMBIO CRÍTICO: Tu red requiere DOS entradas simultáneas
# =====================================================================

# Entrada 1: El cubo CASSI después de pasar por el ShiftBack (31 canales)
dummy_cassi_sb = torch.randn(1, 31, 256, 256) 

# Entrada 2: La fotografía RGB que sirve de guía geométrica (3 canales)
dummy_rgb = torch.randn(1, 3, 256, 256)

# Empaquetamos ambas entradas en una tupla
entradas_dummy = (dummy_cassi_sb, dummy_rgb)

# Le ponemos nombres bonitos a los nodos iniciales y finales para Netron
nombres_entradas = ['Cubo_CASSI_ShiftBack_31ch', 'Imagen_RGB_Guia_3ch']
nombres_salidas = ['Cubo_HSI_Reconstruido_31ch']

# =====================================================================

print("⚙️ Ensamblando el archivo ONNX...")
# Exportamos el grafo
torch.onnx.export(modelo,               
                  entradas_dummy, 
                  "elwrym_abac_test_graph.onnx",   
                  export_params=False,  # False porque solo queremos el mapa, no los pesos    
                  opset_version=11,
                  input_names=nombres_entradas,
                  output_names=nombres_salidas)

print("✅ ¡Listo! Se ha creado el archivo 'elwrym_abac_test_graph.onnx'.")