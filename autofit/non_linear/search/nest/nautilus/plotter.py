import numpy as np

from autofit.plot import SamplesPlotter

class NautilusPlotter(SamplesPlotter):

    def cornerplot(self, **kwargs):
        """
        Plots the `nautilus` plot `cornerplot`, using the package `corner.py`.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """
        import corner
        import matplotlib.pyplot as plt

        points = np.asarray(self.samples.parameter_lists)

        ndim = points.shape[1]

        panelsize = kwargs.get("panelsize") or 3.5
        yticksize = kwargs.get("yticksize") or 16
        xticksize = kwargs.get("xticksize") or 16

        fig, axes = plt.subplots(ndim, ndim, figsize=(panelsize*ndim, panelsize*ndim))

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):

                axes[i,j].tick_params(axis="y", labelsize=yticksize)
                axes[i,j].tick_params(axis="x", labelsize=xticksize)

        corner.corner(
            data=points,
            weights=np.exp(self.samples.weight_list),
            labels=self.model.parameter_labels_with_superscripts_latex,
            fig=fig,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.close()