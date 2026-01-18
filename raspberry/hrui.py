#region ======= Imports =======

from mpque import MPDeque
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from log import print_log, LVL_DBG, LVL_INF, LVL_ERR, DebugLevel

#endregion

#region ======= Global variables =======

# Global plot objects
fig = None
ax = None
anim = None

# Heart rate
hr_estimates = deque(maxlen=100)

# Queue injected by main process
mpdeque = None

def print_log_loc(s: str, log_level: DebugLevel = DebugLevel.INFO):
    s_loc = f"[DSP ] {s}"
    print_log(s_loc, log_level)

#endregion

#region ======= Plot functions =======

def setup_plot():
    """
    Initialize the matplotlib figure and axes.
    This function must be called once before starting the animation.
    """
    global fig, ax

    # Create a new figure
    fig = plt.figure(figsize=(10, 4)) 

    # Create a single subplot (1 row, 1 column)
    ax = fig.add_subplot(1, 1, 1)

    # leave space for text
    plt.subplots_adjust(right=0.82)  

    # Configure Y-axis for heart rate range
    ax.set_ylim(0, 230)

    # No label or ticks on X-axis (time index only)
    ax.set_xlabel("")
    ax.set_xticks([])

    # Y-axis label
    ax.set_ylabel("HR (BPM)")

    # Remove plot borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Big HR text outside the plot
    global hr_text
    hr_text = fig.text(
        0.9, 0.5, "--",
        fontsize=32,
        fontweight="bold",
        ha="center",
        va="center"
    )

def plot_hr(frame_idx):
    """
    Animation callback function.
    Called periodically by FuncAnimation to update the plot.
    """
    global hr_estimates, mpdeque, hr_text

    # Drain mpdequeue
    while True:
        try:
            hr_estimates.append(mpdeque.popright(block=False))
        except IndexError:
            break

    ax.clear()

    # Reapply axis configuration after clear()
    ax.set_ylim(30, 200)
    ax.set_xlabel("")
    ax.set_ylabel("HR (BPM)")
    ax.set_xticks([])

    ax.set_yticks(range(0, 231, 10))
    ax.tick_params(axis="y", labelsize=6)

    # Draw horizontal dashed lines every 10
    for y in range(30, 201, 10):
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.25)

    # Plot HR estimates as red line
    ax.plot(hr_estimates, color='red')

    # Update big HR text outside the plot
    if len(hr_estimates) > 0:
        hr_text.set_text(f"{int(hr_estimates[-1])}\nBPM\n\n{(hr_estimates[-1]/60.0):.2f}\nHz")
    else:
        hr_text.set_text("--")

#endregion

#region ======= UI process =======

def PROC_UI(mpdeque_ui: MPDeque, refresh_ms: int = 1000):
    """
    Start real-time plotting of heart rate estimates.
    """
    global anim, mpdeque

    mpdeque = mpdeque_ui
    
    print_log_loc("Process alive", LVL_INF)

    # Setup plot
    setup_plot()

    # Create animation that updates the plot periodically
    anim = animation.FuncAnimation(
        fig,
        plot_hr,
        interval=refresh_ms,
        cache_frame_data=False
    )

    # Show plot (blocking call)
    plt.show()
    
#endregion
