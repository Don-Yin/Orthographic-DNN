import json
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr


class CorrelationMatrix:
    def __init__(
        self,
        dataframe: pandas.DataFrame,
        method: str = "kendall",
        path_save: Path = None,
        figure_size: tuple = (22, 18),
        adjust_button: float = 0.2,
        adjust_left: float = 0.5,
        which_plot: str = "main",
        human_data_label: str = None,
    ):
        sns.set_theme(font_scale=1.3, style="white")
        self.font_size = 18
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.dataframe = dataframe
        self.method = method
        self.figure_size = figure_size
        self.path_save = path_save
        self.adjust_button = adjust_button
        self.adjust_left = adjust_left
        self.categories = self.analysis_settings["categories"]
        self.categories_model_score = self.analysis_settings["categories_model_score"]
        self.human_data_label = human_data_label

        if type(which_plot) == str:
            getattr(self, which_plot)()
        else:
            getattr(self, which_plot[0])(which_plot[1])

    def main(self):
        # Compute the correlation matrix
        self.corr = self.dataframe.corr(method=self._get_correlation_callable)
        corr_models = self.corr[self.human_data_label].loc[json.load(open(Path("assets", "selected_model_labels.json"), "r")).values()]
        corr_models.to_csv(Path("results", self.analysis_settings["result"], "corr_models.csv"))

        # create a mask dataframe with p* labels
        p_vals = self._get_p_vals()
        p_vals = p_vals - np.eye(*self.corr.shape)
        p_vals = p_vals.applymap(lambda x: "".join(["*" for t in [0.01, 0.05, 0.1] if x <= t]))
        self.corr_with_p: pandas.DataFrame = self.corr.round(2).astype(str) + p_vals

        # Generate a mask for the upper triangle (tril = lower triangle)
        mask = np.triu(np.ones_like(self.corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=self.figure_size)

        # add margin
        f.subplots_adjust(bottom=self.adjust_button, left=self.adjust_left)

        # Generate a custom diverging colormap
        self.cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # set category as the second index
        self.corr["category"] = self.categories
        self.corr_with_p["category"] = self.categories
        self.corr.set_index([self.corr.index, "category"], inplace=True)
        self.corr_with_p.set_index([self.corr_with_p.index, "category"], inplace=True)
        self.corr_with_p = self.corr_with_p.replace({"0\.": "."}, regex=True)

        # Draw the heatmap with the mask and correct aspect ratio
        self.svm = sns.heatmap(
            self.corr,
            mask=mask,
            cmap=self.cmap,
            vmin=0,
            vmax=1,
            center=0,
            square=False,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=self.corr_with_p,
            fmt="",
            annot_kws={"size": self.font_size},
        )

        self.svm.set_xticklabels(self.svm.get_xticklabels(), rotation=90)

        # remove default labels (as to be after the svm is created)
        ax.set_yticklabels(["" for item in ax.get_yticklabels()])
        ax.set_ylabel("")
        self.grouping_label_group_bar_table(ax, df=self.corr)

        # save figure
        self.svm.get_figure().savefig(self.path_save, dpi=400)

    def side(self, name):
        all_types = list(set(self.analysis_settings["categories"]))
        all_types.remove(name)

        # Compute the correlation matrix
        self.corr = self.dataframe.corr(method=self._get_correlation_callable)

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=self.figure_size)

        # add margin
        f.subplots_adjust(bottom=self.adjust_button, left=self.adjust_left)

        # Generate a custom diverging colormap
        self.cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # set category as the second index
        self.corr["category"] = self.categories
        self.corr.set_index([self.corr.index, "category"], inplace=True)
        self.corr = self.corr[[self.human_data_label]]
        self.corr = self.corr.drop(all_types, level=1, axis=0, inplace=False)
        self.corr.reset_index(level=1, drop=True, inplace=True)
        self.corr = self.corr.sort_values(by=[self.human_data_label], ascending=False)

        # create a mask dataframe with p* labels
        p_vals = self._get_p_vals()
        p_vals["category"] = self.categories
        p_vals.set_index([p_vals.index, "category"], inplace=True)
        p_vals = p_vals[[self.human_data_label]]
        p_vals = p_vals.drop(all_types, level=1, axis=0, inplace=False)
        p_vals.reset_index(level=1, drop=True, inplace=True)
        p_vals = p_vals.reindex(list(self.corr.index))
        p_vals = p_vals - np.eye(*self.corr.shape)
        p_vals = p_vals.applymap(lambda x: "".join(["*" for t in [0.01, 0.05, 0.1] if x <= t]))
        self.corr_with_p: pandas.DataFrame = self.corr.round(2).astype(str) + p_vals
        self.corr_with_p = self.corr_with_p.replace({"0\.": "."}, regex=True)
        self.corr_with_p.to_csv(Path("results", self.analysis_settings["result"], "corr_models_with_mask.csv"))

        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        svm = sns.heatmap(
            self.corr,
            mask=None,
            cmap=cmap,
            vmin=0,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=self.corr_with_p,
            fmt="",
            annot_kws={"size": self.font_size},
        )

        svm.get_figure().savefig(Path("results", self.analysis_settings["result"], f"{name}.png"), dpi=400)

    def model_score(self):
        # Compute the correlation matrix
        self.corr = self.dataframe.corr(method=self._get_correlation_callable)

        # create a mask dataframe with p* labels
        p_vals = self._get_p_vals()
        p_vals = p_vals - np.eye(*self.corr.shape)
        p_vals = p_vals.applymap(lambda x: "".join(["*" for t in [0.01, 0.05, 0.1] if x <= t]))
        self.corr_with_p: pandas.DataFrame = self.corr.round(2).astype(str) + p_vals

        # Generate a mask for the upper triangle (tril = lower triangle)
        mask = np.triu(np.ones_like(self.corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=self.figure_size)

        # add margin
        f.subplots_adjust(bottom=self.adjust_button, left=self.adjust_left)

        # Generate a custom diverging colormap
        self.cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # set category as the second index
        self.corr["category"] = self.categories_model_score
        self.corr_with_p["category"] = self.categories_model_score
        self.corr.set_index([self.corr.index, "category"], inplace=True)
        self.corr_with_p.set_index([self.corr_with_p.index, "category"], inplace=True)
        self.corr_with_p = self.corr_with_p.replace({"0\.": "."}, regex=True)

        # Draw the heatmap with the mask and correct aspect ratio
        self.svm = sns.heatmap(
            self.corr,
            mask=mask,
            cmap=self.cmap,
            vmin=self.corr.min().min(),
            vmax=1,
            center=0,
            square=False,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=self.corr_with_p,
            fmt="",
            annot_kws={"size": self.font_size},
        )

        self.svm.set_xticklabels(self.svm.get_xticklabels(), rotation=90)

        # remove default labels (as to be after the svm is created)
        ax.set_yticklabels(["" for item in ax.get_yticklabels()])
        ax.set_ylabel("")
        self.grouping_label_group_bar_table(ax, df=self.corr)

        # save figure
        self.svm.get_figure().savefig(self.path_save, dpi=400)

    def _get_correlation_callable(self, vector_1, vector_2):
        if self.method == "kendall":
            return kendalltau(vector_1, vector_2, variant="c", alternative="greater")[0]
        elif self.method == "spearman":
            return spearmanr(vector_1, vector_2, axis=0, alternative="greater")[0]
        elif self.method == "pearson":
            return pearsonr(vector_1, vector_2)[0]

    def _get_p_vals(self):
        if self.method == "kendall":
            return self.dataframe.corr(method=lambda x, y: kendalltau(x, y, variant="c", alternative="greater")[1])
        elif self.method == "spearman":
            return self.dataframe.corr(method=lambda x, y: spearmanr(x, y, axis=0, alternative="greater")[1])
        elif self.method == "pearson":
            return self.dataframe.corr(method=lambda x, y: pearsonr(x, y)[1])

    def grouping_add_line(self, ax, xpos, ypos):
        # adjust line length here
        line = plt.Line2D([ypos - 0.15, ypos + 0.2], [xpos, xpos], color="black", transform=ax.transAxes)
        line.set_clip_on(False)
        ax.add_line(line)

    def grouping_label_len(self, my_index, level):
        labels = my_index.get_level_values(level)
        return [(k, sum(1 for i in g)) for k, g in groupby(labels)]

    def grouping_label_group_bar_table(self, ax, df):
        xpos = -0.2  # entire label section x position (don't change)
        scale = 1 / df.index.size  # y-axis length (don't change)
        for level in range(df.index.nlevels):
            pos = df.index.size
            for label, rpos in self.grouping_label_len(df.index, level):
                self.grouping_add_line(ax, pos * scale, xpos)
                pos -= rpos
                lypos = (pos + 0.5 * rpos) * scale  # label y position (don't change)
                ax.text(
                    xpos - 0.15, lypos - 0.005, label, ha="left", transform=ax.transAxes
                )  # x position of text; ha: text position in box (0.15)
            self.grouping_add_line(ax, pos * scale, xpos)
            xpos -= 0.27  # distance between categories and labels


if __name__ == "__main__":
    # CorrelationMatrix(dataframe=df, path_save=Path("test_correlation_matrix.png"))
    pass
