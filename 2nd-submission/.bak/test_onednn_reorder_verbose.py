from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from torch.autograd.profiler import record_function


def dump_tensor(tag: str, t: torch.Tensor) -> None:
    channels_last = (
        t.is_contiguous(memory_format=torch.channels_last) if t.dim() == 4 else "n/a"
    )
    print(
        f"{tag}: shape={tuple(t.shape)}, stride={t.stride()}, "
        f"is_contig={t.is_contiguous()}, channels_last={channels_last}"
    )


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_input(memory_format: str) -> torch.Tensor:
    x = torch.randn(8, 3, 224, 224)
    if memory_format == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        x = x.contiguous()
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal oneDNN verbose test to inspect reorder events."
    )
    p.add_argument(
        "--memory-format",
        default="contiguous",
        choices=["contiguous", "channels_last"],
        help="Input tensor memory format.",
    )
    p.add_argument(
        "--verbose-level",
        default="creation",
        choices=["on", "creation"],
        help="oneDNN verbose mode to use around the measured forward pass.",
    )
    p.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Optional output prefix for profiler artifacts.",
    )
    return p.parse_args()


def write_profiler_outputs(prof: profile, save_prefix: str) -> None:
    prefix = Path(save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = prefix.with_suffix(".profiler.txt")
    trace_path = prefix.with_suffix(".chrome_trace.json")
    events_path = prefix.with_suffix(".events.txt")

    summary = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    events = []
    for evt in prof.events():
        if evt.device_type == ProfilerActivity.CPU:
            events.append(
                f"{evt.name}\tself_us={evt.self_cpu_time_total:.3f}\ttotal_us={evt.cpu_time_total:.3f}"
            )

    summary_path.write_text(summary + "\n", encoding="utf-8")
    events_path.write_text("\n".join(events) + "\n", encoding="utf-8")
    prof.export_chrome_trace(str(trace_path))

    print(f"profiler_summary_path={summary_path}")
    print(f"profiler_events_path={events_path}")
    print(f"profiler_trace_path={trace_path}")


def main() -> None:
    args = parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    model = TinyConvNet().eval()
    x = build_input(args.memory_format)

    print(f"torch={torch.__version__}")
    print(f"mkldnn_available={torch.backends.mkldnn.is_available()}")
    print(f"memory_format={args.memory_format}")
    dump_tensor("input", x)

    with torch.no_grad():
        _ = model(x)

    verbose_flag = (
        torch.backends.mkldnn.VERBOSE_ON_CREATION
        if args.verbose_level == "creation"
        else torch.backends.mkldnn.VERBOSE_ON
    )

    print(f"verbose_level={args.verbose_level}")
    print("forward_with_verbose_begin")
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        acc_events=True,
    ) as prof:
        with torch.backends.mkldnn.verbose(verbose_flag):
            with torch.no_grad():
                with record_function("measured_forward"):
                    y = model(x)
    print("forward_with_verbose_end")
    dump_tensor("output", y)

    print("profiler_top_ops_begin")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print("profiler_top_ops_end")

    if args.save_prefix:
        write_profiler_outputs(prof, args.save_prefix)


if __name__ == "__main__":
    main()
