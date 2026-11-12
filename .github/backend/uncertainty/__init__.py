"""Uncertainty analysis package.

The posterior covariance itself lives on MolecularOptimizer
(laplace_approximation, internal_coordinate_posterior), built from the blocks
of the objective the fit minimised. The separate covariance this package used
to compute (core.compute_uncertainty) described a different objective -- no
quantum prior, chi-square rescaling -- and was removed.
"""
