"""Shared runtime helpers for the staged JAX integration work."""

from __future__ import annotations


def ensure_jax_x64():
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX x64 mode is required, but enabling it at runtime did not succeed.")
    return jax


def normalize_platform_name(platform: str | None) -> str | None:
    if platform is None:
        return None
    name = str(platform).lower()
    if name == "gpu":
        return "cuda"
    return name


def resolve_jax_device(platform: str = "default"):
    jax = ensure_jax_x64()
    if platform == "default":
        return None

    normalized = normalize_platform_name(platform)
    matches = [device for device in jax.devices() if normalize_platform_name(device.platform) == normalized]
    if not matches:
        available = ", ".join(f"{device.platform}:{device}" for device in jax.devices())
        raise RuntimeError(
            f"No JAX device matched requested platform {platform!r}. "
            f"Visible devices: {available or 'none'}"
        )
    return matches[0]


def maybe_device_put(value, device):
    if device is None:
        return value
    jax = ensure_jax_x64()
    return jax.device_put(value, device)


def device_label(device) -> str:
    if device is None:
        return "default"
    return f"{device.platform}:{device}"
