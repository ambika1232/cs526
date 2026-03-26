#!/usr/bin/env python3
import math
import argparse


def estimate_transactions(stride: int, elem_size: int = 4, warp_size: int = 32, segment_size: int = 128) -> int:
    if stride <= 0:
        raise ValueError("stride must be positive")
    span_bytes = (((warp_size - 1) * stride) + 1) * elem_size
    return math.ceil(span_bytes / segment_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple warp-level coalescing estimator")
    parser.add_argument("stride", type=int, help="stride across adjacent threads")
    parser.add_argument("--elem-size", type=int, default=4)
    parser.add_argument("--warp-size", type=int, default=32)
    parser.add_argument("--segment-size", type=int, default=128)
    args = parser.parse_args()

    tx = estimate_transactions(args.stride, args.elem_size, args.warp_size, args.segment_size)
    print(f"stride={args.stride} elem_size={args.elem_size} warp_size={args.warp_size} segment_size={args.segment_size}")
    print(f"estimated_transactions_per_warp={tx}")


if __name__ == "__main__":
    main()
