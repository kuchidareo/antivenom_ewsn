import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity


# =========================================================
# 1. Simple CNN: conv3 + fc3
# =========================================================
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)      # [B, 8, 16, 16]
        x = F.relu(x)
        x = self.pool(x)       # [B, 8, 8, 8]

        x = self.conv2(x)      # [B, 16, 8, 8]
        x = F.relu(x)
        x = self.pool(x)       # [B, 16, 4, 4]

        x = self.conv3(x)      # [B, 32, 4, 4]
        x = F.relu(x)

        x = x.flatten(1)       # [B, 512]
        x = self.fc1(x)        # [B, 128]
        x = F.relu(x)

        x = self.fc2(x)        # [B, 64]
        x = F.relu(x)

        x = self.fc3(x)        # [B, 10]
        return x


# =========================================================
# 2. Hook-based capture
# =========================================================
class BackwardOperandCapture:
    """
    Save forward input/output per Conv2d/Linear.
    retain_grad() on module output so output.grad becomes grad_output.
    """
    def __init__(self, model):
        self.saved = {}
        self.handles = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            output.retain_grad()
            self.saved[name] = {
                "module": module,
                "input": x,
                "output": output,
            }
        return hook

    def clear(self):
        self.saved.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =========================================================
# 3. ACD Analyzer
# =========================================================
class ACDAnalyzer:
    def __init__(self, eps=1e-8, flat_block_size=64, md_block_sizes=None):
        self.eps = eps
        self.flat_block_size = flat_block_size
        self.md_block_sizes = md_block_sizes or {
            3: (4, 8, 8),       # e.g. [B,O,I] or [B,I,O]
            4: (2, 8, 8, 16),   # e.g. conv proxy [N,L,C,P]
        }
        self.records = []
        self.prev_masks = {}

    # --------------------------
    # helpers
    # --------------------------
    def _active_mask(self, t):
        return (t.abs() > self.eps)

    def _flatten_runs(self, mask):
        flat = mask.reshape(-1).bool()
        idx = torch.nonzero(flat, as_tuple=False).flatten()
        if idx.numel() == 0:
            return []
        runs = []
        cur = 1
        for k in range(1, idx.numel()):
            if idx[k] == idx[k - 1] + 1:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        return runs

    def _flat_block_stats(self, mask):
        flat = mask.reshape(-1).bool()
        n = flat.numel()
        b = self.flat_block_size
        nonempty = 0
        full = 0
        partial = 0
        for start in range(0, n, b):
            blk = flat[start:start + b]
            s = int(blk.sum().item())
            if s > 0:
                nonempty += 1
                if s == blk.numel():
                    full += 1
                else:
                    partial += 1
        return {
            "flat_nonempty_blocks": nonempty,
            "flat_full_blocks": full,
            "flat_partial_blocks": partial,
            "flat_partial_block_ratio": partial / nonempty if nonempty > 0 else 0.0,
            "flat_full_block_ratio": full / nonempty if nonempty > 0 else 0.0,
        }

    def _multidim_block_stats(self, mask):
        """
        Block occupancy in the tensor's canonical logical shape.
        This is still library-agnostic; it does not assume physical layout.
        """
        mask = mask.bool()
        shape = mask.shape
        ndim = mask.dim()
        if ndim not in self.md_block_sizes:
            return {
                "md_block_shape": None,
                "md_nonempty_blocks": None,
                "md_full_blocks": None,
                "md_partial_blocks": None,
                "md_partial_block_ratio": None,
                "md_full_block_ratio": None,
            }

        block_shape = self.md_block_sizes[ndim]
        if len(block_shape) != ndim:
            raise ValueError(f"Block shape {block_shape} incompatible with ndim={ndim}")

        nonempty = 0
        full = 0
        partial = 0

        ranges = [range(0, shape[d], block_shape[d]) for d in range(ndim)]

        # nested loops without itertools for readability
        if ndim == 3:
            for i0 in ranges[0]:
                for i1 in ranges[1]:
                    for i2 in ranges[2]:
                        blk = mask[
                            i0:i0 + block_shape[0],
                            i1:i1 + block_shape[1],
                            i2:i2 + block_shape[2],
                        ]
                        s = int(blk.sum().item())
                        if s > 0:
                            nonempty += 1
                            if s == blk.numel():
                                full += 1
                            else:
                                partial += 1

        elif ndim == 4:
            for i0 in ranges[0]:
                for i1 in ranges[1]:
                    for i2 in ranges[2]:
                        for i3 in ranges[3]:
                            blk = mask[
                                i0:i0 + block_shape[0],
                                i1:i1 + block_shape[1],
                                i2:i2 + block_shape[2],
                                i3:i3 + block_shape[3],
                            ]
                            s = int(blk.sum().item())
                            if s > 0:
                                nonempty += 1
                                if s == blk.numel():
                                    full += 1
                                else:
                                    partial += 1

        return {
            "md_block_shape": block_shape,
            "md_nonempty_blocks": nonempty,
            "md_full_blocks": full,
            "md_partial_blocks": partial,
            "md_partial_block_ratio": partial / nonempty if nonempty > 0 else 0.0,
            "md_full_block_ratio": full / nonempty if nonempty > 0 else 0.0,
        }

    def _jaccard(self, a, b):
        a = a.reshape(-1).bool()
        b = b.reshape(-1).bool()
        if a.numel() != b.numel():
            return None
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        return inter / union if union > 0 else 1.0

    def _summarize_mask(self, acd_mask):
        total = acd_mask.numel()
        active = int(acd_mask.sum().item())
        density = active / total if total > 0 else 0.0

        runs = self._flatten_runs(acd_mask)
        run_count = len(runs)
        avg_run_length = sum(runs) / run_count if run_count > 0 else 0.0
        max_run_length = max(runs) if run_count > 0 else 0
        fragmentation = run_count / active if active > 0 else 0.0

        flat_blk = self._flat_block_stats(acd_mask)
        md_blk = self._multidim_block_stats(acd_mask)

        out = {
            "active_count": active,
            "total_count": total,
            "density": density,
            "run_count": run_count,
            "avg_run_length": avg_run_length,
            "max_run_length": max_run_length,
            "fragmentation": fragmentation,
        }
        out.update(flat_blk)
        out.update(md_blk)
        return out

    # --------------------------
    # Linear ACD
    # --------------------------
    def linear_dW_acd(self, x, grad_output):
        # x: [B, I], grad_output: [B, O]
        x_mask = self._active_mask(x)
        dy_mask = self._active_mask(grad_output)
        return dy_mask[:, :, None] & x_mask[:, None, :]   # [B, O, I]

    def linear_dX_acd(self, weight, grad_output):
        # weight: [O, I], grad_output: [B, O]
        w_mask = self._active_mask(weight)
        dy_mask = self._active_mask(grad_output)
        return dy_mask[:, :, None] & w_mask[None, :, :]   # [B, O, I]
        # logically this represents interaction tuples (b,o,i)

    # --------------------------
    # Conv2d ACD
    # --------------------------
    def _conv_unfold_input(self, module, x):
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        dh, dw = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)

        x_col = F.unfold(
            x,
            kernel_size=(kh, kw),
            dilation=(dh, dw),
            padding=(ph, pw),
            stride=(sh, sw),
        )  # [N, P, L]
        return x_col

    def conv2d_dW_acd(self, module, x, grad_output):
        """
        dW[cout,p] = sum_{n,l} dY[n,cout,l] * X_col[n,p,l]
        acd: [N, L, Cout, P]
        """
        x_col = self._conv_unfold_input(module, x)                # [N, P, L]
        n, p, l = x_col.shape
        dy = grad_output.reshape(n, grad_output.shape[1], -1)     # [N, Cout, L]

        x_mask = self._active_mask(x_col).transpose(1, 2)         # [N, L, P]
        dy_mask = self._active_mask(dy).transpose(1, 2)           # [N, L, Cout]

        return dy_mask[:, :, :, None] & x_mask[:, :, None, :]     # [N, L, Cout, P]

    def conv2d_dX_acd(self, module, grad_output):
        """
        Proxy for dX:
        each active output grad at (n,l,cout) interacts with all active kernel entries W[cout,p].
        acd: [N, L, Cout, P]
        """
        weight = module.weight.detach()                           # [Cout, Cin, Kh, Kw]
        cout = weight.shape[0]
        p = weight[0].numel()

        w_mask = self._active_mask(weight.reshape(cout, p))       # [Cout, P]

        n = grad_output.shape[0]
        dy = grad_output.reshape(n, grad_output.shape[1], -1)     # [N, Cout, L]
        dy_mask = self._active_mask(dy).transpose(1, 2)           # [N, L, Cout]

        return dy_mask[:, :, :, None] & w_mask[None, None, :, :]  # [N, L, Cout, P]

    # --------------------------
    # public API
    # --------------------------
    def analyze_module(self, name, module, x, grad_output, step):
        records = []

        if isinstance(module, nn.Linear):
            acd_dw = self.linear_dW_acd(x, grad_output)
            rec_dw = self._make_record(
                name=name,
                kind="linear_dW",
                acd_mask=acd_dw,
                step=step,
                prev_key=(name, "dW"),
            )
            records.append(rec_dw)

            acd_dx = self.linear_dX_acd(module.weight.detach(), grad_output)
            rec_dx = self._make_record(
                name=name,
                kind="linear_dX",
                acd_mask=acd_dx,
                step=step,
                prev_key=(name, "dX"),
            )
            records.append(rec_dx)

        elif isinstance(module, nn.Conv2d):
            acd_dw = self.conv2d_dW_acd(module, x, grad_output)
            rec_dw = self._make_record(
                name=name,
                kind="conv2d_dW",
                acd_mask=acd_dw,
                step=step,
                prev_key=(name, "dW"),
            )
            records.append(rec_dw)

            acd_dx = self.conv2d_dX_acd(module, grad_output)
            rec_dx = self._make_record(
                name=name,
                kind="conv2d_dX",
                acd_mask=acd_dx,
                step=step,
                prev_key=(name, "dX"),
            )
            records.append(rec_dx)

        self.records.extend(records)
        return records

    def _make_record(self, name, kind, acd_mask, step, prev_key):
        summary = self._summarize_mask(acd_mask)

        prev = self.prev_masks.get(prev_key, None)
        stability = self._jaccard(prev, acd_mask.cpu()) if prev is not None else None
        self.prev_masks[prev_key] = acd_mask.detach().cpu()

        return {
            "step": step,
            "name": name,
            "kind": kind,
            "shape": tuple(acd_mask.shape),
            "stability_jaccard": stability,
            **summary,
        }

    def print_step(self, step):
        print(f"\n===== ACD Step {step} =====")
        for r in self.records:
            if r["step"] == step:
                print(
                    f"{r['name']:>5s} | {r['kind']:>9s} | "
                    f"shape={r['shape']} | "
                    f"density={r['density']:.6f} | "
                    f"frag={r['fragmentation']:.6f} | "
                    f"avg_run={r['avg_run_length']:.2f} | "
                    f"flat_partial={r['flat_partial_block_ratio']:.4f} | "
                    f"md_partial={r['md_partial_block_ratio']} | "
                    f"stability={r['stability_jaccard']}"
                )

    def get_records(self):
        return self.records


# =========================================================
# 4. Profiler utilities
# =========================================================
def collect_profiler_summary(model, x, target, criterion, optimizer, device):
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

    # key_averages gives operator-level timing
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    return prof, table


def extract_backwardish_events(prof):
    rows = []
    for evt in prof.key_averages():
        key = evt.key
        if ("backward" in key.lower()) or ("grad" in key.lower()) or ("convolution_backward" in key.lower()):
            rows.append({
                "key": key,
                "self_cpu_time_total": getattr(evt, "self_cpu_time_total", None),
                "cpu_time_total": getattr(evt, "cpu_time_total", None),
                "self_cuda_time_total": getattr(evt, "self_cuda_time_total", None),
                "cuda_time_total": getattr(evt, "cuda_time_total", None),
                "count": getattr(evt, "count", None),
            })
    return rows


# =========================================================
# 5. Full demo
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = SmallCNN().to(device)
    capture = BackwardOperandCapture(model)
    analyzer = ACDAnalyzer(
        eps=1e-8,
        flat_block_size=64,
        md_block_sizes={
            3: (4, 8, 16),      # for linear ACD [B,O,I] or [B,O,I]
            4: (2, 8, 8, 16),   # for conv ACD [N,L,C,P]
        },
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    steps = 5
    batch_size = 16

    profiler_tables = []
    profiler_backward_rows = []

    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        capture.clear()

        x = torch.randn(batch_size, 1, 16, 16, device=device)

        # vary sparsity pattern across steps
        sparsity = 0.30 + 0.10 * (step % 3)
        mask = (torch.rand_like(x) > sparsity)
        x = x * mask

        target = torch.randint(0, 10, (batch_size,), device=device)

        # profile only last 2 steps to reduce overhead in toy code
        do_profile = (step >= steps - 2)

        if do_profile:
            prof, table = collect_profiler_summary(model, x, target, criterion, optimizer, device)
            profiler_tables.append((step, table))
            profiler_backward_rows.append((step, extract_backwardish_events(prof)))

            # run once more for ACD capture because profiler function used separate pass
            optimizer.zero_grad(set_to_none=True)
            capture.clear()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        else:
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        # analyze module-wise ACD using captured inputs and grad_outputs
        for name, saved in capture.saved.items():
            module = saved["module"]
            x_in = saved["input"]
            grad_output = saved["output"].grad.detach()
            analyzer.analyze_module(
                name=name,
                module=module,
                x=x_in,
                grad_output=grad_output,
                step=step,
            )

        analyzer.print_step(step)

    # -----------------------------------------
    # print profiler summaries
    # -----------------------------------------
    print("\n===== PROFILER TOP OPS =====")
    for step, table in profiler_tables:
        print(f"\n--- Step {step} ---")
        print(table)

    print("\n===== BACKWARD-LIKE PROFILER ROWS =====")
    for step, rows in profiler_backward_rows:
        print(f"\n--- Step {step} ---")
        for r in rows:
            print(r)

    print("\n===== FINAL ACD RECORDS =====")
    for r in analyzer.get_records():
        print(r)

    capture.close()


if __name__ == "__main__":
    main()
