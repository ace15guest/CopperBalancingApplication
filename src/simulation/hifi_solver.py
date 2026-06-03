from src.models import CopperDensityMap, SimResult, Stackup


def solve_hifi(stackup: Stackup, density_maps: list[CopperDensityMap], delta_temp_c: float) -> SimResult:
    """
    High-fidelity solver.
    Extends CLT with lamination profile, via density, and material asymmetry effects.
    """
    raise NotImplementedError
