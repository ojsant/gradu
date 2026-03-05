import datetime as dt
from dataclasses import dataclass, field

# TODO: dataclasses -> ABCs
# from abc import ABC, abstractmethod


@dataclass
class SoloConstants:
    name: str = "SolO"
    sectors: list[str] = field(default_factory=lambda: ["sun", "asun", "north", "south"])
    mission_start: dt.datetime = dt.datetime(2021, 1, 1)
    mission_end: dt.datetime = dt.datetime(2025, 12, 31)
    native_cadence: str = "1min"
    pitch_angle_mu_columns: list[str] = field(
        default_factory=lambda: [f"Pitch_Angle_{dir}" for dir in ["S", "A", "N", "D"]]
        )
    pitch_angle_sigma_columns: list[str] = field(
        default_factory=lambda: [f"Pitch_Angle_Sigma_{dir}" for dir in ["S", "A", "N", "D"]]
        )
    bin_width_deg: float = 30.0


@dataclass
class WindConstants:
    name: str = "Wind"
    sectors: list[str] = field(default_factory=lambda: [f"P{i}" for i in range(8)])
    mission_start: dt.datetime = dt.datetime(2005, 1, 1)
    mission_end: dt.datetime = dt.datetime(2025, 12, 31)
    native_cadence: str = "12s"
    bin_width_deg: float = 22.5
