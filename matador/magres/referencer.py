# coding: utf-8
# Distributed under the terms of the MIT license.

from typing import Dict, List, Optional
import numpy as np

from matador.plotting.plotting import plotting_function
from matador.crystal import Crystal

__all__ = ("MagresReferencer", )


class MagresReferencer:
    """ Class for referencing NMR predictions with experimental data. """

    def __init__(
        self,
        structures_exp: List[Crystal],
        shifts_exp: List[Dict[str, List[float]]],
        species: str,
        structures: Optional[List[Crystal]] = None,
    ):
        self.structures_exp = structures_exp
        self.shifts_exp = shifts_exp
        self.species = species
        self.structures = structures

        self._calc_shifts = []
        self._expt_shifts = []
        self._fit_weights = []
        self._fit_structures = []
        self._fitted = False

        for structure, shifts in zip(self.structures_exp, self.shifts_exp):
            self.match_exp_structure_shifts(structure, shifts[self.species])

        self.fit()

        if self.structures is not None:
            self.structures = self.set_shifts_from_fit(self.structures)

    def match_exp_structure_shifts(self, structure, shifts):
        relevant_sites = [site for site in structure if site.species == self.species]
        calc_shifts = sorted(
            [site["chemical_shielding_iso"] for site in relevant_sites]
        )
        _shifts = shifts
        if (
            len(_shifts) <= len(relevant_sites)
            and len(relevant_sites) % len(_shifts) == 0
        ):
            multiplier = len(relevant_sites) // len(_shifts)
            _shifts = [shift for cell in [_shifts] * multiplier for shift in cell]
            _weights = [1 / multiplier for shift in _shifts]
        else:
            raise RuntimeError(
                f"Incompatible shift sizes: {len(relevant_sites)} (theor.) vs {len(_shifts)} (expt.), "
                "please provide commensurate cells."
            )

        _shifts = sorted(_shifts, reverse=True)

        self._calc_shifts.extend(calc_shifts)
        self._expt_shifts.extend(_shifts)
        self._fit_weights.extend(_weights)
        self._fit_structures.extend(len(_shifts) * [structure.formula_tex])

        return _shifts, _weights, calc_shifts

    def set_shifts_from_fit(self, structures):
        for ind, struc in enumerate(structures):
            for jnd, site in enumerate(struc):
                if site.species == self.species:
                    structures[ind][jnd]["chemical_shift_iso"] = self.predict(site["chemical_shielding_iso"])

        return structures

    def fit(self):
        import statsmodels.api as sm

        _fit_shifts = sm.add_constant(self._calc_shifts)
        self.fit_model = sm.regression.linear_model.OLS(self._expt_shifts, _fit_shifts)
        self.fit_results = self.fit_model.fit()
        self.fit_intercept = self.fit_results.params[0]
        self.fit_gradient = self.fit_results.params[1]
        self.fit_rsquared = self.fit_results.rsquared
        self._fitted = True

    def predict(self, shifts):
        _shifts = np.asarray(shifts)
        return self.fit_gradient * _shifts + self.fit_intercept

    def print_fit_summary(self):
        if self._fitted:
            print("Performed OLS fit for: δ_expt = m * δ_calc + c")
            print(f"m = {self.fit_gradient:3.3f} ± {self.fit_results.bse[1]:3.3f}")
            print(f"c = {self.fit_intercept:3.3f} ± {self.fit_results.bse[0]:3.3f} ppm")
            print(f"R² = {self.fit_rsquared:3.3f}.")
        else:
            raise RuntimeError("Fit has not yet been performed.")

    @plotting_function
    def plot_fit(self, padding=100):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_ylim(np.min(self._calc_shifts) - padding, np.max(self._calc_shifts) + padding)
        ax.set_xlim(np.min(self._expt_shifts) - padding, np.max(self._expt_shifts) + padding)
        ax = sns.regplot(
            y=self._calc_shifts,
            x=self._expt_shifts,
            scatter=False,
            ax=ax,
            color="grey",
            truncate=False,
        )
        sns.scatterplot(
            y=self._calc_shifts,
            x=self._expt_shifts,
            hue=self._fit_structures,
            palette="Dark2",
            ax=ax,
            zorder=1e10,
        )
        ax.set_xlabel("$\\delta_\\mathrm{expt}$ (ppm)")
        ax.set_ylabel("$\\sigma_\\mathrm{calc}$ (ppm)")
