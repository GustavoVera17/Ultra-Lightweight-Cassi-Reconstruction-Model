import torch
from sir_cnn_W32SB_full import SIR_CNN

print("Generando modelo 3D/Gráfico de HRNet-W32...")
modelo = SIR_CNN()

# SIR_CNN solo recibe el cubo CASSI disperso
dummy_cassi = torch.randn(1, 1, 256, 286) 

# Exportamos a ONNX
torch.onnx.export(modelo,               
                  dummy_cassi, 
                  "mi_red_hrnet.onnx",   
                  export_params=False,      
                  opset_version=11)

print("¡Listo! Arrastra el archivo 'mi_red_hrnet.onnx' a la página netron.app")