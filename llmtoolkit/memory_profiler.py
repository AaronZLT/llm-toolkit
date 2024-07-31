from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler._memory_profiler import _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX

from .utils import (
    print_rank_0,
)


def export_memory_timeline_html(
    self, path, device, figsize=(20, 12), title=None
) -> None:
    """Exports the memory timeline as an HTML file which contains
    the memory timeline plot embedded as a PNG file."""
    # Check if user has matplotlib installed, return gracefully if not.
    import importlib.util

    matplotlib_spec = importlib.util.find_spec("matplotlib")
    if matplotlib_spec is None:
        print(
            "export_memory_timeline_html failed because matplotlib was not found."
        )
        return

    from base64 import b64encode
    from os import remove
    from tempfile import NamedTemporaryFile

    import matplotlib.pyplot as plt
    import numpy as np

    mt = self._coalesce_timeline(device)
    times, sizes = np.array(mt[0]), np.array(mt[1])
    # For this timeline, start at 0 to match Chrome traces.
    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / 1024**3

    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()

    print_rank_0("Processing memory trace, to find the memory overhead of each category, such as parameter, gradient, etc. The output is rounded to 3 decimals.")
    category_max_memory = {}
    total_memory_overhead = 0
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        max_value = round(np.max(stacked[:, i + 1] - stacked[:, i]), 3)
        category_max_memory[category] = max_value
        total_memory_overhead += max_value
        print_rank_0(f"Max memory for {category}: {max_value} GB")
    print_rank_0(f"Total memory overhead: {total_memory_overhead} GB")

    # Plot memory timeline as stacked data
    fig = plt.figure(figsize=figsize, dpi=80)
    axes = fig.gca()
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        axes.fill_between(
            times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
        )
    fig.legend(
        [f"Unknown {category_max_memory[i]} GB" if i is None else f"{i.name} {category_max_memory[i]} GB" for i in _CATEGORY_TO_COLORS])
    # Usually training steps are in magnitude of ms.
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Memory (GB)")
    title = "\n\n".join(
        ([title] if title else [])
        + [
            f"Max memory allocated: {max_memory_allocated/(10**9):.2f} GB \n"
            f"Max memory reserved: {max_memory_reserved/(10**9):.2f} GB\n"
            f"Total memory overhead: {total_memory_overhead:.3f} GB"
        ]
    )
    axes.set_title(title)

    # Embed the memory timeline image into the HTML file
    tmpfile = NamedTemporaryFile("wb", suffix=".png", delete=False)
    tmpfile.close()
    fig.savefig(tmpfile.name, format="png")

    try:
        fig.savefig(path.replace(".html", ".png"))
    except:
        print_rank_0(
            "Memory overhead cannot be saved into .png (path is unspecified)! HTML is not impacted.")

    with open(tmpfile.name, "rb") as tmp:
        encoded = b64encode(tmp.read()).decode("utf-8")
        html = f"""<html>
<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
<body>
<img src='data:image/png;base64,{encoded}'>
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)
    remove(tmpfile.name)
