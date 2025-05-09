


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc


def plot_inductance_over_gun_flux(shots, gun_fluxes, toi):
    sbc=Sbc()
    df = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    
    for s in range(len(shots)):
    
    
    return
    
    
    
def calc_nearest_time(times, time):
    """
    Find the value in 'times' that is closest to 'time'.

    Parameters:
        times (array-like): Array of time values.
        time (float): Target time.

    Returns:
        float: The nearest value in 'times' to 'time'.
    """
    times = np.asarray(times)
    idx = np.abs(times - time).argmin()
    return times[idx]