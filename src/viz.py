import base64
import io

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from src.models import COLOR_MAP, GRID


def viz_grid(
    grid: list[list[int]], color_map: dict[int, str], ax: Axes | None = None
) -> Axes:
    """
    Visualizes a grid of integer cells as colored squares on a given matplotlib Axes.

    Each integer in the grid is mapped to a color defined in color_map.
    A slight grey border is drawn between cells.

    Parameters:
        grid (list[list[int]]): A 2D list representing the grid of integers.
        color_map (dict[int, str]): A mapping from integer values to color strings.
                                    Example: {0: 'white', 1: 'blue', 2: 'red'}
        ax (plt.Axes, optional): An Axes object to plot on. If None, a new figure and axis are created.

    Returns:
        plt.Axes: The Axes with the plotted grid.
    """
    # Make a local copy of the grid and convert to a NumPy array.
    grid = grid.copy()
    grid_np = np.array(grid)
    rows, cols = grid_np.shape

    # Establish an ordering for the colormap based on sorted keys.
    ordered_keys = sorted(color_map.keys())
    mapping = {val: idx for idx, val in enumerate(ordered_keys)}

    # Map grid values to indices.
    mapped_grid = np.vectorize(mapping.get)(grid_np)

    # Create a ListedColormap and a BoundaryNorm for crisp cell boundaries.
    cmap = mcolors.ListedColormap([color_map[val] for val in ordered_keys])
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(ordered_keys) + 0.5, 1), len(ordered_keys)
    )

    # Create an axis if not provided.
    if ax is None:
        fig, ax = plt.subplots()

    # Display the grid.
    ax.imshow(mapped_grid, cmap=cmap, norm=norm)

    # Set tick positions to align gridlines with cell boundaries.
    ax.set_xticks([x - 0.5 for x in range(1 + cols)])
    ax.set_yticks([y - 0.5 for y in range(1 + rows)])

    # Draw gridlines with a slight grey border between cells.
    ax.grid(which="both", color="white", linewidth=3)

    # Remove tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Hide the axes spines for a cleaner look.
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def viz_many(
    grids: list[list[GRID]], row_border_colors: list[str], color_map: dict[int, str]
) -> None:
    """
    Visualizes multiple grids arranged in rows as specified by a 2D list.

    Each inner list in 'grids' represents one row of subplots.
    For example, given:
        grids = [[G1, G2], [G3, G4], [G5]]
    The function creates three rows of subplots:
      - Row 1: G1 and G2 side by side.
      - Row 2: G3 and G4 side by side.
      - Row 3: G5 is placed in the first column and the remaining subplot(s) are hidden.

    Additionally, each row is outlined with a border whose color is taken from
    row_border_colors (a list of hex color strings, one per row).

    Parameters:
        grids (list[list[list[int]]]): A 2D list where each element is a grid (a 2D list of integers).
        row_border_colors (list[str]): A list of hex color strings. Its length must equal the number of rows.
        color_map (dict[int, str]): A mapping from integer values to color strings.
    """
    n_rows = len(grids)
    n_cols = max(len(row) for row in grids) if grids else 0

    if len(row_border_colors) != n_rows:
        raise ValueError(
            f"Expected {n_rows} row border colors, but got {len(row_border_colors)}."
        )

    # Create a grid of subplots.
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), constrained_layout=True
    )

    # Normalize axs into a 2D array even if there's only one row or column.
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    # Plot each grid in its corresponding subplot.
    for i, row in enumerate(grids):
        for j in range(n_cols):
            if j < len(row):
                viz_grid(row[j], color_map, ax=axs[i, j])
            else:
                # Hide any unused subplots.
                axs[i, j].axis("off")

    # Force the layout to update so that we get the correct positions.
    fig.canvas.draw()

    # For each row, compute the union of the axes positions and draw a colored border.
    for i in range(n_rows):
        # Get positions (in figure coordinates) for all axes in the i-th row.
        row_positions = [axs[i, j].get_position() for j in range(n_cols)]
        left = min(pos.x0 for pos in row_positions)
        right = max(pos.x1 for pos in row_positions)
        bottom = min(pos.y0 for pos in row_positions)
        top = max(pos.y1 for pos in row_positions)
        width = right - left
        height = top - bottom

        # Create a rectangle patch with no fill and the specified border color.
        rect = Rectangle(
            (left, bottom),
            width,
            height,
            fill=False,
            edgecolor=row_border_colors[i],
            lw=5,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.add_artist(rect)

    plt.show()


def base64_from_grid(grid: GRID) -> str:
    """
    Converts a grid to a base64-encoded PNG image.

    Parameters:
        grid (GRID): A 2D list representing the grid of integers.

    Returns:
        str: Base64-encoded string of the grid visualization as a PNG image.
    """
    # Create a figure and axis with no padding
    fig, ax = plt.subplots(figsize=(6, 6))

    # Remove all margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Visualize the grid using the existing viz_grid function
    viz_grid(grid, COLOR_MAP, ax=ax)

    # Save the figure to a BytesIO buffer with no padding
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", pad_inches=0, dpi=150)
    buffer.seek(0)

    # Convert to base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Close the figure to free memory
    plt.close(fig)

    return image_base64
