"""Comparable pre-action occupancy slices, installed routes and physical tracks."""
import numpy as np


def plot(path, grid, box, anchor, agent, traces, maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    colors = ListedColormap(["#edf2f1", "#c2cbd0", "#455964"])
    lo, hi = box
    sx, sy, sz = grid["bounds"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    names = ("r123_shared", "ch5_batch_seed", "ch5_r123_seed")
    titles = ("R123 shared map (archived poses)", "CH5 local map (batch seed)", "CH5 local map (R123 seed)")
    visible = [lo, hi, anchor]
    for rows in traces.values():
        visible.extend(r["position"] for r in rows)
        visible.append(rows[-1]["next_position"])
        visible.extend(rows[-1]["route"][rows[-1]["route_index"]:])
    low = np.maximum(np.min(visible, axis=0)-1.25, 0)
    high = np.minimum(np.max(visible, axis=0)+1.25, [sx, sy, sz])
    for col, (name, title) in enumerate(zip(names, titles)):
        rows = traces[name]
        last = rows[-1]
        occupied = maps[name][-1][:, :, last["map"]["impact_voxel"][2]]
        classes = np.where(occupied <= grid["free_threshold"], 0,
                           np.where(occupied >= grid["occupied_threshold"], 2, 1))
        xy, xz = axes[:, col]
        xy.imshow(classes.T, cmap=colors, vmin=0, vmax=2, origin="lower",
                  extent=(0, sx, 0, sy), interpolation="nearest")
        positions = np.asarray([rows[0]["position"], *[r["next_position"] for r in rows]])
        planned = np.asarray([last["position"], *last["route"][last["route_index"]:]])
        for ax, y in ((xy, 1), (xz, 2)):
            ax.add_patch(Rectangle((lo[0], lo[y]), hi[0]-lo[0], hi[y]-lo[y], fill=False,
                                   edgecolor="#bb423e", linewidth=1.4, label="Obstacle 0 (truth outline)"))
            ax.plot(planned[:, 0], planned[:, y], "--o", color="#da880f", ms=3, lw=1.8, label="Remaining plan")
            ax.plot(positions[:, 0], positions[:, y], color="#006d8f", lw=2.0, label="Actual trajectory")
            ax.scatter(*np.asarray(last["waypoint"])[[0, y]], marker="D", color="#ad6909", s=24, label="Tracking waypoint")
            ax.scatter(anchor[0], anchor[y], marker="x", color="#b6292b", s=55, label="Archived CH5 impact")
            ax.set(xlim=(low[0], high[0]), ylim=(low[y], high[y]), xlabel="x (m)", ylabel=("y (m)" if y == 1 else "z (m)"),
                   aspect="equal")
        xy.set_title(f"{title}\nBefore action {last['step']} · impact voxel {last['map']['impact_voxel_state']}")
    fig.suptitle(f"{path.parent.name} · vehicle {agent} · XY occupancy at impact height, XZ projection below\n"
                 "Light = free, gray = unknown, dark = occupied. Tracks include the last action.")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False)
    try:
        fig.savefig(path, dpi=145)
    finally:
        plt.close(fig)
