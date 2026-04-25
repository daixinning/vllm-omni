# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Tensors are streamed chunk-by-chunk with simple request-response protocol:
Producer writes a chunk, waits for consumer ACK, then writes next chunk.
This ensures only one chunk exists in /dev/shm at a time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from multiprocessing import shared_memory

import numpy as np
import torch
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.data import DiffusionOutput

logger = init_logger(__name__)

# Minimum tensor size to use SHM instead of inline transfer
_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB

# Default maximum size for a single SHM segment (64MB)
_DEFAULT_SHM_MAX_SEGMENT_SIZE = 64 * 1024 * 1024
_SHM_ACK_TIMEOUT = 30  # seconds to wait for ACK between SHM chunks


def _get_max_shm_segment_size() -> int:
    """Get max chunk size for SHM transfer.

    Since only one chunk exists in SHM at a time (write -> wait ACK -> unlink),
    we can use up to 90% of available /dev/shm space. Environment variable
    override takes precedence for containerized environments.
    """
    if envs.VLLM_SHM_MAX_SEGMENT_SIZE is not None:
        return envs.VLLM_SHM_MAX_SEGMENT_SIZE

    try:
        stat = os.statvfs("/dev/shm")
        available = stat.f_frsize * stat.f_bavail
        # Use 90% of available, leave 10% for other processes
        return int(available * 0.9)
    except Exception as e:
        logger.warning("Failed to get /dev/shm space: %s, using default 64MB", e)

    return _DEFAULT_SHM_MAX_SEGMENT_SIZE


def _tensor_to_bytes(tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    flat = tensor.view(torch.uint8).reshape(-1).numpy()
    return flat, flat.nbytes


def _bytes_to_tensor(data: bytes | np.ndarray, torch_dtype: str, shape: list) -> torch.Tensor:
    dtype = getattr(torch, torch_dtype.replace("torch.", ""))
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype)
    return t.reshape(shape)


def _should_use_shm(tensor: torch.Tensor) -> bool:
    """Check if a tensor should be transferred via SHM instead of inline."""
    if not isinstance(tensor, torch.Tensor):
        return False
    return tensor.nelement() * tensor.element_size() > _SHM_TENSOR_THRESHOLD


def pack_diffusion_output_shm(output: object, result_mq: MessageQueue, ack_mq: MessageQueue | None = None) -> None:
    """Send a output through result_mq, streaming large tensors
    chunk-by-chunk via SHM.

    Small tensors (<= _SHM_TENSOR_THRESHOLD) are sent inline with the output
    object. Large tensors are streamed via SHM with chunk-by-chunk protocol.

    Protocol:
        1. Send output object (small tensors inline, large tensors set to None)
        2. Send header with field descriptors (only for SHM fields)
        3. For each SHM chunk:
           - Create SHM, write data, send chunk info
           - Wait for ACK from consumer
           - Unlink SHM
    """
    diff_output = output if isinstance(output, DiffusionOutput) else getattr(output, "result", None)
    if not isinstance(diff_output, DiffusionOutput):
        # Not a DiffusionOutput, send directly without SHM
        result_mq.enqueue(output)
        result_mq.enqueue({"__shm_fields__": []})
        return

    max_chunk_size = _get_max_shm_segment_size()

    # Collect tensor fields and decide which ones need SHM
    tensor_fields = []
    for field_name in ("output", "trajectory_latents", "trajectory_timesteps", "trajectory_log_probs"):
        val = getattr(diff_output, field_name, None)
        if isinstance(val, torch.Tensor):
            if _should_use_shm(val):
                # Large tensor: will use SHM, clear from output
                tensor_fields.append((field_name, val, True))  # (name, tensor, use_shm)
                setattr(diff_output, field_name, None)
            else:
                # Small tensor: keep inline, no SHM needed
                tensor_fields.append((field_name, val, False))

    # Build header with only SHM field descriptors
    shm_fields = []
    for field_name, tensor, use_shm in tensor_fields:
        if use_shm:
            tensor_cpu = tensor.detach().cpu().contiguous()
            nbytes = tensor_cpu.nelement() * tensor_cpu.element_size()
            num_chunks = (nbytes + max_chunk_size - 1) // max_chunk_size
            shm_fields.append(
                {
                    "field": field_name,
                    "shape": list(tensor_cpu.shape),
                    "torch_dtype": str(tensor_cpu.dtype),
                    "total_nbytes": nbytes,
                    "num_chunks": num_chunks,
                    "_tensor_cpu": tensor_cpu,
                }
            )

    # Send output and header
    header = {"__shm_fields__": [{k: v for k, v in fd.items() if k != "_tensor_cpu"} for fd in shm_fields]}
    result_mq.enqueue(output)
    result_mq.enqueue(header)

    # If no SHM fields, we're done
    if not shm_fields:
        return

    # Stream each tensor chunk-by-chunk: write -> wait ACK -> unlink.
    for fd in shm_fields:
        flat, total_nbytes = _tensor_to_bytes(fd["_tensor_cpu"])
        offset = 0

        for i in range(fd["num_chunks"]):
            chunk_size = min(max_chunk_size, total_nbytes - offset)
            shm = None

            try:
                # Create SHM and write chunk
                shm = shared_memory.SharedMemory(create=True, size=chunk_size)
                np.copyto(
                    np.ndarray((chunk_size,), dtype=np.uint8, buffer=shm.buf),
                    flat[offset : offset + chunk_size],
                )
                shm.close()

                # Send chunk info
                result_mq.enqueue(
                    {
                        "__shm_chunk__": True,
                        "name": shm.name,
                        "size": chunk_size,
                        "field": fd["field"],
                        "chunk_index": i,
                    }
                )

                # Wait for ACK before proceeding
                if ack_mq is not None:
                    ack_mq.dequeue(timeout=_SHM_ACK_TIMEOUT)
                    logger.debug("Received ACK for %s chunk %d", fd["field"], i)

                # Unlink SHM after consumer has read it
                try:
                    shared_memory.SharedMemory(name=shm.name).unlink()
                except FileNotFoundError:
                    # Already unlinked by consumer
                    pass

            except Exception as e:
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
                logger.warning(
                    "SHM alloc failed (%d bytes): %s, falling back to inline for %s chunk %d",
                    chunk_size,
                    e,
                    fd["field"],
                    i,
                )
                result_mq.enqueue(
                    {
                        "__shm_chunk__": True,
                        "__inline__": True,
                        "data": bytes(flat[offset : offset + chunk_size]),
                        "size": chunk_size,
                        "field": fd["field"],
                        "chunk_index": i,
                    }
                )
                if ack_mq is not None:
                    ack_mq.dequeue(timeout=_SHM_ACK_TIMEOUT)
            offset += chunk_size


_DEQUEUE_TIMEOUT_S = 5.0


def _dequeue_with_failure_check(mq, timeout, is_failed_fn=None):
    import zmq

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if deadline is None:
            chunk_t = _DEQUEUE_TIMEOUT_S
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("dequeue timed out")
            chunk_t = min(_DEQUEUE_TIMEOUT_S, remaining)
        try:
            return mq.dequeue(timeout=chunk_t)
        except (TimeoutError, zmq.error.Again):
            if is_failed_fn is not None and is_failed_fn():
                raise EngineDeadError()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("dequeue timed out")


def unpack_diffusion_output_shm(
    result_mq: MessageQueue,
    ack_mq: MessageQueue | None = None,
    timeout: float | None = None,
    is_failed_fn: Callable[[], bool] | None = None,
) -> object:
    """Receive a output from result_mq, reassembling streamed tensors.

    Small tensors are received inline with the output object.
    Large tensors are reassembled from SHM chunks.

    This function handles all dequeue operations internally, making it symmetric
    with pack_diffusion_output_shm which handles all enqueue operations.

    Protocol:
        1. Dequeue response object
        2. Dequeue header with SHM field descriptors
        3. For each SHM field:
           - For each chunk:
             - Dequeue chunk info
             - Read data from SHM
             - Close SHM
             - Send ACK to producer
           - Reconstruct tensor and set on output

    Args:
        result_mq: Result message queue to dequeue from
        ack_mq: Optional ACK message queue for flow control
        timeout: Optional timeout in seconds for the initial response dequeue.
            If None, blocks indefinitely.

    Returns:
        The received object (DiffusionOutput or other type)
    """
    # Dequeue response and header
    response = _dequeue_with_failure_check(result_mq, timeout, is_failed_fn)
    header = _dequeue_with_failure_check(result_mq, _SHM_ACK_TIMEOUT, is_failed_fn)

    # Check if header indicates any SHM fields
    if not isinstance(header, dict):
        return response

    shm_fields = header.get("__shm_fields__", [])
    if not shm_fields:
        # No SHM fields, all data is inline
        return response

    diff_output = response if isinstance(response, DiffusionOutput) else getattr(response, "result", None)
    if not isinstance(diff_output, DiffusionOutput):
        return response

    # Process each SHM field
    for fd in shm_fields:
        buf = bytearray(fd["total_nbytes"])
        offset = 0

        for chunk_idx in range(fd["num_chunks"]):
            chunk_msg = _dequeue_with_failure_check(result_mq, _SHM_ACK_TIMEOUT, is_failed_fn)
            if not isinstance(chunk_msg, dict) or not chunk_msg.get("__shm_chunk__"):
                raise RuntimeError(f"Expected SHM chunk, got: {type(chunk_msg)}")

            # Inline fallback: data sent directly via MessageQueue
            size = chunk_msg["size"]
            if chunk_msg.get("__inline__"):
                buf[offset : offset + size] = chunk_msg["data"]
                logger.debug("Received inline chunk %d (%d bytes) for %s", chunk_idx, size, fd["field"])
            else:
                shm = shared_memory.SharedMemory(name=chunk_msg["name"])
                try:
                    buf[offset : offset + size] = shm.buf[:size]
                    logger.debug("Received SHM chunk %d (%d bytes) for %s", chunk_idx, size, fd["field"])
                finally:
                    shm.close()

            offset += size
            if ack_mq is not None:
                ack_mq.enqueue({"status": "chunk_processed"})

        setattr(diff_output, fd["field"], _bytes_to_tensor(buf, fd["torch_dtype"], fd["shape"]))

    return response
