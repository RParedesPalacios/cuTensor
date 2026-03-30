#!/usr/bin/env python3
"""
Benchmark de multiplicacion de matrices en GPU:
- PyTorch (CUDA)
- cuTensor

Genera:
- CSV con medidas por tamano
- Informe Markdown con tabla y resumen automatico
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import numpy as np
import torch
from cuTensor import cuTensor as CuTensor


@dataclass
class BenchResult:
    size: int
    cutensor_mean_ms: float
    cutensor_std_ms: float
    torch_mean_ms: float
    torch_std_ms: float
    speedup_cutensor_vs_torch: float
    winner: str
    max_abs_error: float
    max_rel_error: float


def parse_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError("Todos los tamanos deben ser enteros positivos.")
        sizes.append(value)
    if not sizes:
        raise ValueError("No se proporcionaron tamanos validos.")
    return sizes


def time_backend(
    fn: Callable[[], object],
    repeats: int,
    device: int,
) -> tuple[float, float]:
    timings_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        timings_ms.append(float(start.elapsed_time(end)))
        del out
    return mean(timings_ms), pstdev(timings_ms)


def warmup(fn: Callable[[], object], steps: int, device: int) -> None:
    for _ in range(steps):
        out = fn()
        torch.cuda.synchronize(device)
        del out


def compare_outputs(
    a_cutensor: CuTensor,
    b_cutensor: CuTensor,
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
    device: int,
) -> tuple[float, float]:
    c_cutensor = CuTensor.mm(a_cutensor, b_cutensor)
    torch.cuda.synchronize(device)
    c_cutensor_np = c_cutensor.to_numpy()
    del c_cutensor

    c_torch = torch.matmul(a_torch, b_torch)
    torch.cuda.synchronize(device)
    c_torch_np = c_torch.detach().cpu().numpy()
    del c_torch

    diff = np.abs(c_cutensor_np - c_torch_np)
    max_abs = float(np.max(diff))
    denom = np.maximum(np.abs(c_torch_np), 1e-8)
    max_rel = float(np.max(diff / denom))
    return max_abs, max_rel


def run_benchmark(
    sizes: list[int],
    repeats: int,
    warmup_steps: int,
    device: int,
    seed: int,
) -> list[BenchResult]:
    rng = np.random.default_rng(seed)
    results: list[BenchResult] = []

    for size in sizes:
        print(f"[benchmark] Tamano {size}x{size}")
        a_np = rng.standard_normal((size, size), dtype=np.float32)
        b_np = rng.standard_normal((size, size), dtype=np.float32)

        a_cutensor = CuTensor.from_numpy(a_np, device=device, name=f"A_{size}")
        b_cutensor = CuTensor.from_numpy(b_np, device=device, name=f"B_{size}")

        a_torch = torch.from_numpy(a_np).to(device=device)
        b_torch = torch.from_numpy(b_np).to(device=device)

        cutensor_fn = lambda: CuTensor.mm(a_cutensor, b_cutensor)
        torch_fn = lambda: torch.matmul(a_torch, b_torch)

        warmup(cutensor_fn, warmup_steps, device)
        warmup(torch_fn, warmup_steps, device)

        cutensor_mean_ms, cutensor_std_ms = time_backend(cutensor_fn, repeats, device)
        torch_mean_ms, torch_std_ms = time_backend(torch_fn, repeats, device)
        max_abs_error, max_rel_error = compare_outputs(
            a_cutensor, b_cutensor, a_torch, b_torch, device
        )

        speedup = torch_mean_ms / cutensor_mean_ms if cutensor_mean_ms > 0 else math.inf
        if speedup > 1.0:
            winner = "cuTensor"
        elif speedup < 1.0:
            winner = "PyTorch"
        else:
            winner = "Empate"

        results.append(
            BenchResult(
                size=size,
                cutensor_mean_ms=cutensor_mean_ms,
                cutensor_std_ms=cutensor_std_ms,
                torch_mean_ms=torch_mean_ms,
                torch_std_ms=torch_std_ms,
                speedup_cutensor_vs_torch=speedup,
                winner=winner,
                max_abs_error=max_abs_error,
                max_rel_error=max_rel_error,
            )
        )

        del a_cutensor
        del b_cutensor
        del a_torch
        del b_torch
        torch.cuda.empty_cache()

    return results


def write_csv(path: Path, results: list[BenchResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "size",
                "cutensor_mean_ms",
                "cutensor_std_ms",
                "torch_mean_ms",
                "torch_std_ms",
                "speedup_cutensor_vs_torch",
                "winner",
                "max_abs_error",
                "max_rel_error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.size,
                    f"{r.cutensor_mean_ms:.6f}",
                    f"{r.cutensor_std_ms:.6f}",
                    f"{r.torch_mean_ms:.6f}",
                    f"{r.torch_std_ms:.6f}",
                    f"{r.speedup_cutensor_vs_torch:.6f}",
                    r.winner,
                    f"{r.max_abs_error:.6e}",
                    f"{r.max_rel_error:.6e}",
                ]
            )


def write_markdown_report(
    path: Path,
    results: list[BenchResult],
    args: argparse.Namespace,
) -> None:
    speedups = [r.speedup_cutensor_vs_torch for r in results]
    geo_mean_speedup = math.exp(mean(math.log(x) for x in speedups if x > 0))
    cutensor_wins = sum(1 for r in results if r.winner == "cuTensor")
    torch_wins = sum(1 for r in results if r.winner == "PyTorch")
    ties = len(results) - cutensor_wins - torch_wins

    best = max(results, key=lambda r: r.speedup_cutensor_vs_torch)
    worst = min(results, key=lambda r: r.speedup_cutensor_vs_torch)

    half = max(1, len(results) // 2)
    small_avg = mean(r.speedup_cutensor_vs_torch for r in results[:half])
    large_avg = mean(r.speedup_cutensor_vs_torch for r in results[half:])

    if large_avg > small_avg:
        trend = "cuTensor mejora su ventaja a medida que crece el tamano."
    elif large_avg < small_avg:
        trend = "cuTensor pierde ventaja relativa en tamanos grandes."
    else:
        trend = "No se aprecia una tendencia clara por tamano."

    lines: list[str] = []
    lines.append("# Informe de benchmark: PyTorch CUDA vs cuTensor")
    lines.append("")
    lines.append(f"- Fecha: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- GPU: `{torch.cuda.get_device_name(args.device)}`")
    lines.append(f"- CUDA (torch): `{torch.version.cuda}`")
    lines.append(f"- Repeticiones por tamano: `{args.repeats}`")
    lines.append(f"- Warmup por backend: `{args.warmup}`")
    lines.append(f"- Tamanos: `{args.sizes}`")
    lines.append("")
    lines.append(
        "Las medidas usan eventos CUDA y sincronizacion explicita del dispositivo "
        "tras cada operacion."
    )
    lines.append("")
    lines.append("## Resultados por tamano")
    lines.append("")
    lines.append(
        "| Tamano | cuTensor mean (ms) | PyTorch mean (ms) | Speedup cuTensor vs PyTorch | Ganador | Max abs err | Max rel err |"
    )
    lines.append(
        "|---:|---:|---:|---:|:---|---:|---:|"
    )
    for r in results:
        lines.append(
            f"| {r.size}x{r.size} | {r.cutensor_mean_ms:.3f} | {r.torch_mean_ms:.3f} | "
            f"{r.speedup_cutensor_vs_torch:.3f}x | {r.winner} | "
            f"{r.max_abs_error:.3e} | {r.max_rel_error:.3e} |"
        )

    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- Victorias cuTensor: **{cutensor_wins}**")
    lines.append(f"- Victorias PyTorch: **{torch_wins}**")
    lines.append(f"- Empates: **{ties}**")
    lines.append(f"- Speedup geometrico medio (cuTensor vs PyTorch): **{geo_mean_speedup:.3f}x**")
    lines.append(
        f"- Mejor caso cuTensor: **{best.size}x{best.size}** ({best.speedup_cutensor_vs_torch:.3f}x)"
    )
    lines.append(
        f"- Peor caso cuTensor: **{worst.size}x{worst.size}** ({worst.speedup_cutensor_vs_torch:.3f}x)"
    )
    lines.append(f"- Tendencia: {trend}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara multiplicacion de matrices en CUDA entre PyTorch y cuTensor."
    )
    parser.add_argument(
        "--sizes",
        default="128,256,512,1024,2048",
        help="Lista de tamanos NxN separada por comas. Ej: 256,512,1024",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Numero de repeticiones medidas por tamano y backend.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Numero de iteraciones de calentamiento por backend y tamano.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Indice de GPU CUDA a usar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Semilla para generar matrices aleatorias.",
    )
    parser.add_argument(
        "--outdir",
        default="benchmark/results",
        help="Directorio de salida para CSV y reporte Markdown.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = parse_sizes(args.sizes)

    if args.repeats <= 0:
        raise ValueError("--repeats debe ser > 0")
    if args.warmup < 0:
        raise ValueError("--warmup debe ser >= 0")

    if not torch.cuda.is_available():
        print("ERROR: CUDA no esta disponible en PyTorch.", file=sys.stderr)
        return 1

    if args.device < 0 or args.device >= torch.cuda.device_count():
        print(
            f"ERROR: device {args.device} fuera de rango. GPUs visibles: {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        return 1

    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"matmul_benchmark_{timestamp}.csv"
    report_path = outdir / f"matmul_report_{timestamp}.md"

    print(f"[info] Ejecutando benchmark en GPU {args.device}: {torch.cuda.get_device_name(args.device)}")
    results = run_benchmark(
        sizes=sizes,
        repeats=args.repeats,
        warmup_steps=args.warmup,
        device=args.device,
        seed=args.seed,
    )
    write_csv(csv_path, results)
    write_markdown_report(report_path, results, args)

    print(f"[ok] CSV: {csv_path}")
    print(f"[ok] Informe: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
