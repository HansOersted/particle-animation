import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# User inputs
# The values can represent percentages, concentrations,
# or relative amounts, as long as they describe the ratio.
# =========================================================
monomer_amount = 80
hmw_amount = 15
lmw_amount = 5

# Total number of particles shown on the screen.
# These particles are only for visualization and do not
# represent the real number of molecules.
total_display_particles = 240

# Animation settings
frames = 500
interval_ms = 40
box_size = 10.0


# =========================================================
# Helper functions
# =========================================================
def allocate_particle_counts(amounts, total_particles=200):
    """
    Allocate the number of displayed particles according
    to the ratio of Monomer, HMW, and LMW.
    """
    total_amount = sum(amounts.values())

    if total_amount <= 0:
        raise ValueError("The total amount must be greater than zero.")

    counts = {}
    raw_counts = {}

    for name, value in amounts.items():
        raw_counts[name] = total_particles * value / total_amount

    # Round the particle numbers
    for name, value in raw_counts.items():
        counts[name] = int(round(value))

    # Correct possible rounding errors
    difference = total_particles - sum(counts.values())
    names = list(amounts.keys())

    fractional_parts = {
        name: raw_counts[name] - int(raw_counts[name])
        for name in names
    }

    if difference > 0:
        sorted_names = sorted(
            names,
            key=lambda x: fractional_parts[x],
            reverse=True
        )

        for i in range(difference):
            counts[sorted_names[i % len(sorted_names)]] += 1

    elif difference < 0:
        sorted_names = sorted(
            names,
            key=lambda x: fractional_parts[x]
        )

        i = 0

        while difference < 0 and i < 10000:
            name = sorted_names[i % len(sorted_names)]

            if counts[name] > 0:
                counts[name] -= 1
                difference += 1

            i += 1

    return counts


def initialize_particles(number, particle_size, speed, box_size):
    """
    Initialize random positions and velocities for particles.
    """

    positions = np.random.rand(number, 2) * box_size

    angles = np.random.rand(number) * 2 * np.pi

    velocities = np.column_stack(
        (np.cos(angles), np.sin(angles))
    ) * speed

    return {
        "positions": positions,
        "velocities": velocities,
        "particle_size": particle_size,
        "base_speed": speed
    }


def update_particle_motion(
    particle_data,
    noise_strength=0.03,
    speed_limit_factor=2.0
):
    """
    Update particle positions and velocities for one animation frame.
    """

    positions = particle_data["positions"]
    velocities = particle_data["velocities"]
    base_speed = particle_data["base_speed"]

    # Add a small random disturbance to make the motion less artificial
    velocities += np.random.randn(*velocities.shape) * noise_strength

    # Limit the maximum particle speed
    speeds = np.linalg.norm(
        velocities,
        axis=1,
        keepdims=True
    )

    maximum_speed = base_speed * speed_limit_factor

    clipped_speeds = np.clip(
        speeds,
        1e-8,
        maximum_speed
    )

    velocities = (
        velocities /
        np.maximum(speeds, 1e-8) *
        clipped_speeds
    )

    # Update particle positions
    positions += velocities

    # Reflect particles when they hit the boundaries
    for dimension in [0, 1]:

        lower_boundary = positions[:, dimension] < 0

        positions[lower_boundary, dimension] = (
            -positions[lower_boundary, dimension]
        )

        velocities[lower_boundary, dimension] *= -1

        upper_boundary = positions[:, dimension] > box_size

        positions[upper_boundary, dimension] = (
            2 * box_size -
            positions[upper_boundary, dimension]
        )

        velocities[upper_boundary, dimension] *= -1

    particle_data["positions"] = positions
    particle_data["velocities"] = velocities


# =========================================================
# Prepare the particle ratios
# =========================================================
amounts = {
    "Monomer": monomer_amount,
    "HMW": hmw_amount,
    "LMW": lmw_amount
}

particle_counts = allocate_particle_counts(
    amounts,
    total_display_particles
)


# =========================================================
# Visual properties of the three particle types
#
# Monomer:
#   Medium particle size
#   Medium motion speed
#
# HMW:
#   Large particle size
#   Slow motion
#
# LMW:
#   Small particle size
#   Fast motion
# =========================================================
particle_properties = {

    "Monomer": {
        "color": "tab:blue",
        "marker": "o",
        "size": 80,
        "speed": 0.08
    },

    "HMW": {
        "color": "tab:red",
        "marker": "s",
        "size": 180,
        "speed": 0.035
    },

    "LMW": {
        "color": "tab:green",
        "marker": "^",
        "size": 35,
        "speed": 0.11
    }
}


# =========================================================
# Initialize all particles
# =========================================================
particles = {}

for name, number in particle_counts.items():

    properties = particle_properties[name]

    particles[name] = initialize_particles(
        number=number,
        particle_size=properties["size"],
        speed=properties["speed"],
        box_size=box_size
    )


# =========================================================
# Create the animation window
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect("equal")

ax.set_xticks([])
ax.set_yticks([])

title = (
    "Monomer / HMW / LMW Particle Visualization\n"
    f"Input amounts: "
    f"Monomer={monomer_amount}, "
    f"HMW={hmw_amount}, "
    f"LMW={lmw_amount}\n"
    f"Displayed particles: "
    f"Monomer={particle_counts['Monomer']}, "
    f"HMW={particle_counts['HMW']}, "
    f"LMW={particle_counts['LMW']}"
)

ax.set_title(title, fontsize=11)


# =========================================================
# Add an information box
# =========================================================
information_text = ax.text(
    0.02,
    0.02,
    "",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)


# =========================================================
# Create scatter plots for each particle type
# =========================================================
scatter_plots = {}

for name in ["Monomer", "HMW", "LMW"]:

    properties = particle_properties[name]

    scatter_plots[name] = ax.scatter(
        particles[name]["positions"][:, 0],
        particles[name]["positions"][:, 1],
        s=properties["size"],
        c=properties["color"],
        marker=properties["marker"],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
        label=name
    )

ax.legend(loc="upper right")


# =========================================================
# Animation update function
# =========================================================
def update(frame):

    # Update the motion of every particle type
    for name in particles:
        update_particle_motion(particles[name])

    # Update particle positions in the plot
    for name in scatter_plots:
        scatter_plots[name].set_offsets(
            particles[name]["positions"]
        )

    information_text.set_text(
        f"Frame: {frame}\n"
        "Monomer: medium size, medium speed\n"
        "HMW: large size, slow motion\n"
        "LMW: small size, fast motion"
    )

    return list(scatter_plots.values()) + [information_text]


# =========================================================
# Create and display the animation
# =========================================================
animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=interval_ms,
    blit=False
)

plt.tight_layout()
plt.show()


# =========================================================
# Optional: save the animation as an MP4 file
#
# FFmpeg must be installed on the computer.
# =========================================================
# animation.save(
#     "monomer_hmw_lmw_animation.mp4",
#     writer="ffmpeg",
#     fps=1000 // interval_ms
# )