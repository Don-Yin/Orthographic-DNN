import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import seaborn as sns


class RidgePlot:
    def __init__(
        self,
        dataframe: pandas.DataFrame,
        colname_group: str,
        colname_variable: str,
        name_save: str,
        hue_level: int,
        path_save: Path,
        means: list[float] = None,
        draw_density: bool = True,
        whether_double_extreme_lines: bool = False,
        x_lim: tuple[float] = None,
    ):
        self.x_lim = x_lim
        self.name_save = name_save
        self.dataframe = dataframe
        self.colname_group = colname_group
        self.colname_variable = colname_variable
        self.hue_level = hue_level
        self.path_save = path_save
        self.means = means
        self.draw_density = draw_density
        self.whether_double_extreme_lines = whether_double_extreme_lines
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.line_color = "k" if self.analysis_settings["prime_data_folder"] == "prime_data_normal" else "r"
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0)})  # change back to "white" if no grid
        self.init_facetgrid_obj()
        self.main()

    def init_facetgrid_obj(self):
        if self.analysis_settings["prime_data_folder"] == "prime_data_normal":
            self.g = sns.FacetGrid(
                self.dataframe,
                row=self.colname_group,
                hue=self.colname_group,
                aspect=15,
                height=0.5,
                palette=sns.cubehelix_palette(
                    self.hue_level, rot=-0.2, light=0.6, dark=0.6, hue=0
                ),  # adjust main color here
            )
        elif self.analysis_settings["prime_data_folder"] == "prime_data_position_corrected":
            self.g = sns.FacetGrid(
                self.dataframe,
                row=self.colname_group,
                hue=self.colname_group,
                aspect=15,
                height=0.5,
                palette=sns.cubehelix_palette(self.hue_level, rot=-0.2, light=0.6, dark=0.6, hue=0),
            )

    def main(self):
        # Draw the densities in a few steps
        if self.draw_density:
            self.g.map(sns.kdeplot, self.colname_variable, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
            self.g.map(sns.kdeplot, self.colname_variable, clip_on=False, color="w", lw=2, bw_adjust=0.5)

        # Draw mean line (double line width is line position is either max or min of means)
        if not self.means:
            self.g.map(lambda i, **kw: plt.axvline(i.mean(), color=self.line_color, linewidth=1.2), self.colname_variable)

        else:
            for ax, pos in zip(self.g.axes.flat, self.means):
                line_width = 1.2
                line_width = (
                    2 * line_width
                    if pos in [min(self.means), max(self.means)] and self.whether_double_extreme_lines
                    else line_width
                )
                ax.axvline(x=pos, color=self.line_color, linestyle="-", linewidth=line_width)

        # passing color=None to refline() uses the hue mapping
        self.g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(-0.17, 0.2, label, fontweight="bold", color="k", ha="left", va="center", transform=ax.transAxes)

        self.g.map(label, self.colname_variable)  # put label

        # Set the subplots to overlap
        self.g.figure.subplots_adjust(hspace=0)

        # Remove axes details that don't play well with overlap
        self.g.set_titles("")
        self.g.set(yticks=[], ylabel="")
        if self.x_lim:
            self.g.set(xlim=self.x_lim)
        self.g.despine(bottom=True, left=True)

        self.g.figure.savefig(self.path_save / self.name_save)


if __name__ == "__main__":

    #  dummy data
    rs = np.random.RandomState(1979)
    x = rs.randn(500)  # Value
    g = np.tile(list("ABCDEFGHIJ"), 50)  # Group id
    df = pandas.DataFrame(dict(s=x, group=g))
    m = df.group.map(ord)
    df["s"] += m

    RidgePlot(df, colname_group="group", colname_variable="s", name_save="test.png")
