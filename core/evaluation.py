from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from common.EvaluationMetrics import evaluate_stream


# 推理产物按数组分别落盘为 .npy，而不是打包成 npz。
# 原因：npz 需要先把整份数据写一遍，再读一遍，13 GB 量级的复制既慢又吃磁盘；
# 单独的 .npy 可以直接 np.load(mmap_mode="r")，指标计算全程磁盘映射。
ARRAY_NAMES = ("prediction", "target", "persistence", "aux_target")


def _open_memmap(path: Path, shape, dtype=np.float32, mode: str = "w+") -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        str(path),
        mode=mode,
        dtype=dtype,
        shape=tuple(int(v) for v in shape),
    )


def _store_chunk(target: np.ndarray, offset: int, values: np.ndarray) -> int:
    count = int(values.shape[0])
    stop = offset + count
    target[offset:stop] = values
    return stop


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dump_dir: str | Path,
    batches_per_flush: int = 16,
) -> Dict[str, np.memmap]:
    """推理一次并把结果流式写入磁盘映射文件。

    旧实现把每个 batch 的结果 append 到列表，最后 ``np.concatenate`` 成完整张量。
    测试集 17,485 x 12 x 71 x 73 的单数组就是 4.35 GB，prediction/target/
    persistence 三个叠加 13 GB，再经反标准化复制一份，峰值接近 48 GB，进程被
    内核 SIGKILL(137)。

    这里预先在磁盘上分配 memmap，逐块写入，内存只保留当前 batch 的输出。
    """
    model.eval()
    dump_root = Path(dump_dir)
    dump_root.mkdir(parents=True, exist_ok=True)

    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__"):
        raise ValueError(
            "run_inference needs a DataLoader backed by a sized dataset so the "
            "disk-backed arrays can be preallocated"
        )
    total = int(len(dataset))

    iterator = iter(loader)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("DataLoader produced no batches; nothing to evaluate") from exc

    if total <= 0:
        raise ValueError("dataset reported zero samples")

    probe_tec, probe_aux_in, _, probe_aux_gt = first
    probe_tec = probe_tec.float().to(device)
    probe_aux = probe_aux_in.float().to(device)
    probe_out = model(probe_tec, probe_aux)
    if isinstance(probe_out, tuple):
        probe_out = probe_out[0]

    # Shapes come from the actual tensors, never from config guesses:
    # ``aux_in`` is the *input* window while ``aux_target`` is the *output* one.
    steps_out = int(probe_out.shape[1])
    aux_steps = int(probe_aux_gt.shape[1])
    aux_dim = int(probe_aux_gt.shape[-1])
    height, width = int(probe_out.shape[-2]), int(probe_out.shape[-1])

    arrays = {
        "prediction": _open_memmap(
            dump_root / "prediction.npy", (total, steps_out, height, width)
        ),
        "target": _open_memmap(
            dump_root / "target.npy", (total, steps_out, height, width)
        ),
        "persistence": _open_memmap(
            dump_root / "persistence.npy", (total, steps_out, height, width)
        ),
        "aux_target": _open_memmap(
            dump_root / "aux_target.npy", (total, aux_steps, aux_dim)
        ),
    }

    def consume(batch, offset: int) -> int:
        tec_in, aux_in, tec_gt, aux_gt = batch
        tec_in = tec_in.float().to(device)
        aux_in = aux_in.float().to(device)
        out = model(tec_in, aux_in)
        if isinstance(out, tuple):
            out = out[0]
        baseline = tec_in[:, -1:, :, :].repeat(1, out.shape[1], 1, 1)

        stop = _store_chunk(arrays["prediction"], offset, out.detach().cpu().numpy())
        _store_chunk(arrays["target"], offset, tec_gt.numpy())
        _store_chunk(
            arrays["persistence"], offset, baseline.detach().cpu().numpy()
        )
        _store_chunk(arrays["aux_target"], offset, aux_gt.numpy())
        return stop

    offset = consume(first, 0)
    pending = 1
    for batch in iterator:
        offset = consume(batch, offset)
        pending += 1
        if pending >= max(1, int(batches_per_flush)):
            for array in arrays.values():
                array.flush()
            pending = 0

    if offset != total:
        raise RuntimeError(
            f"streamed {offset} samples but the dataset reported {total}; the "
            "DataLoader may be dropping or repeating samples"
        )

    for array in arrays.values():
        array.flush()
    return arrays


def load_predictions(path: str | Path) -> Dict[str, np.memmap]:
    """以只读磁盘映射方式打开一份推理产物目录。"""
    root = Path(path)
    loaded: Dict[str, np.memmap] = {}
    for name in ARRAY_NAMES:
        file_path = root / f"{name}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"missing prediction array: {file_path}")
        loaded[name] = np.load(file_path, mmap_mode="r")
    return loaded


def evaluate_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    persistence: Optional[np.ndarray] = None,
    latitude_bands: int = 6,
    sample_chunk: int = 256,
) -> Dict[str, Any]:
    """对流式/磁盘映射数组计算完整指标，峰值内存与样本数无关。"""
    return evaluate_stream(
        prediction,
        target,
        persistence,
        bands=latitude_bands,
        sample_chunk=sample_chunk,
    )


def save_metrics(metrics: Dict[str, Any], path: str | Path) -> Path:
    import json

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
            ensure_ascii=False,
            default=lambda value: (
                value.item() if hasattr(value, "item") else str(value)
            ),
        )
    return target_path
