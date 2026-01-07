import time
import torch
from thop import profile, clever_format
import torch.nn
from datasets.transforms import *
import time
from model import LossWrapper, CrowdSatNet, load_model
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")


def build_model():

    model = CrowdSatNet(num_classes=2)
    model.to(device)
    return model


def measure_flops_and_params(model, input_shape, device):
    dummy_input = torch.randn(*input_shape, device=device)
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    return flops_str, params_str


def measure_latency(model, input_shape, device, warmup=10, runs=50):
    dummy_input = torch.randn(*input_shape, device=device)
    model.eval()

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)  # ms
            else:
                t0 = time.perf_counter()
                _ = model(dummy_input)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0  # ms
            times_ms.append(elapsed_ms)

    avg_ms = sum(times_ms) / len(times_ms)
    return avg_ms


def measure_peak_memory(model, input_shape, device):
    if device.type != "cuda":
        print("Peak GPU memory measurement only works on CUDA devices.")
        return None

    dummy_input = torch.randn(*input_shape, device=device)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(dummy_input)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / 1024.0 ** 2
    return peak_mb


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 3, 256, 256)  # batch_size, C, H, W

    model = build_model().to(device)

    # 1) FLOPs & Params
    flops_str, params_str = measure_flops_and_params(model, input_shape, device)
    print(f"FLOPs (per {input_shape[-1]}x{input_shape[-2]} patch): {flops_str}")
    print(f"Params: {params_str}")

    # 2) Latency
    avg_ms = measure_latency(model, input_shape, device, warmup=10, runs=50)
    print(f"Average latency: {avg_ms:.3f} ms "
          f"(≈ {avg_ms / 1000.0:.4f} s) per {input_shape[-1]}x{input_shape[-2]} patch")

    # 3) Peak GPU memory
    peak_mb = measure_peak_memory(model, input_shape, device)
    if peak_mb is not None:
        print(f"Peak GPU memory (forward): {peak_mb:.2f} MB")
