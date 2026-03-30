# Benchmark: PyTorch CUDA vs cuTensor

Este directorio incluye un benchmark para comparar multiplicacion de matrices (`matmul`) en GPU entre:

- `PyTorch` con CUDA
- `cuTensor` (esta libreria)

## Script

- `matmul_benchmark.py`

## Requisitos

- `cuTensor` instalado y funcional en el entorno Python.
- `torch` con soporte CUDA.
- GPU NVIDIA visible desde el sistema.

## Ejecucion

Desde la raiz del repo:

```bash
python3 benchmark/matmul_benchmark.py \
  --sizes 128,256,512,1024,2048 \
  --repeats 20 \
  --warmup 5 \
  --device 0 \
  --order alternate
```

## Salidas

Se generan en `benchmark/results/`:

- `matmul_benchmark_<timestamp>.csv`: resultados crudos por tamano.
- `matmul_report_<timestamp>.md`: informe con tabla y resumen automatico.

## Que compara exactamente

- Para cada tamano `N`, genera matrices aleatorias `N x N` en `float32`.
- Carga los datos una vez en cada backend (cuTensor y PyTorch).
- Prealoca la salida en ambos backends (`cuTensor.mm_out` y `torch.mm(..., out=...)`).
- Mide solo la operacion `matmul` (con sincronizacion CUDA explicita tras cada iteracion y sin incluir asignacion de salida).
- Valida precision numerica comparando salida cuTensor vs PyTorch (`max abs error` y `max rel error`).
- Puedes elegir el orden de ejecucion con `--order`:
  - `alternate` (recomendado para reducir sesgo de orden)
  - `cutensor_first`
  - `pytorch_first`

## Notas

- El reporte usa `speedup_cutensor_vs_torch = tiempo_torch / tiempo_cutensor`.
  - `> 1.0`: cuTensor mas rapido.
  - `< 1.0`: PyTorch mas rapido.
