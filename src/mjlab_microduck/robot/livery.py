"""Cosmetic Microduck livery. Material rgba only — never mass, inertia, collision, or actuators."""

from __future__ import annotations

import mujoco

_LAVENDER = (0.72, 0.58, 0.84, 1.0)
_TEAL = (0.55, 0.80, 0.88, 1.0)
_YELLOW = (0.96, 0.82, 0.15, 1.0)

LAVENDER_LIVERY: dict[str, tuple[float, float, float, float]] = {
    "right_shell_material": _LAVENDER,
    "left_shell_material": _LAVENDER,
    "upper_leg_left_material": _LAVENDER,
    "upper_leg_right_material": _LAVENDER,
    "top_head_shell_material": _LAVENDER,
    "sole_left_material": _LAVENDER,
    "sole_right_material": _LAVENDER,
    "roller_blade_material": _LAVENDER,
    "noenoeil_material": _TEAL,
    "jaw_material": _YELLOW,
    "jaw_soft_material": _YELLOW,
    "soft_mouth_top_material": _YELLOW,
    "bottom_head_shell_material": _YELLOW,
    "foot_left_material": _YELLOW,
    "foot_right_material": _YELLOW,
    "ankle_left_material": _YELLOW,
    "ankle_right_material": _YELLOW,
    "ankle_l_v1_material": _YELLOW,
    "ankle_r_v1_material": _YELLOW,
}


def apply_lavender_livery(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    for mat in spec.materials:
        rgba = LAVENDER_LIVERY.get(mat.name)
        if rgba is None:
            for key, value in LAVENDER_LIVERY.items():
                if mat.name.endswith(key):
                    rgba = value
                    break
        if rgba is not None:
            mat.rgba = rgba
    return spec
