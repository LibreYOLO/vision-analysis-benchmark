"""Build-time and boot-time assertions for the RF100-VL campaign image.

Runs twice: once during `docker build` (CPU only, so CUDA checks are skipped)
and again on the rented box, where the GPU assertions do apply. A wrong build
must fail here rather than three hours into a campaign.
"""

import sys

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{'OK  ' if condition else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    require_gpu = "--require-gpu" in sys.argv

    import torch

    check("torch imports", True, f"{torch.__version__}")
    check("torch is a CUDA build", torch.version.cuda is not None, str(torch.version.cuda))

    import libreyolo
    from libreyolo.validation.config import ValidationConfig

    check("libreyolo imports", True, libreyolo.__version__)

    # The two knobs the protocol depends on. Without eval_max_det every AP we
    # produce would silently be AP at maxDets 100 instead of 500.
    fields = getattr(ValidationConfig, "__dataclass_fields__", {})
    check("ValidationConfig.eval_max_det", "eval_max_det" in fields)

    from libreyolo.utils.amp import normalize_amp_dtype

    check("amp_dtype plumbing", normalize_amp_dtype("bfloat16") is not None)

    import pycocotools  # noqa: F401
    import va_bench
    from va_bench.rf100vl_data import long_path  # noqa: F401

    check("va_bench imports", True, getattr(va_bench, "__version__", "unknown"))

    import rf100vl  # noqa: F401

    check("rf100vl package imports", True)

    if require_gpu:
        check("CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            check("GPU visible", True, f"{name} sm_{major}{minor}")
            # sm_120 (Blackwell) needs a cu128 build; catch a mismatched image
            # here rather than at the first kernel launch.
            arches = torch.cuda.get_arch_list()
            check(
                f"sm_{major}{minor} in torch arch list",
                any(f"sm_{major}{minor}" in a for a in arches),
                ",".join(arches),
            )
            probe = torch.randn(1024, 1024, device="cuda")
            check("cuda matmul", torch.isfinite(probe @ probe).all().item())

    if FAILURES:
        print(f"\nIMAGE_VERIFY_FAILED: {', '.join(FAILURES)}")
        return 1
    print("\nIMAGE_VERIFY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
