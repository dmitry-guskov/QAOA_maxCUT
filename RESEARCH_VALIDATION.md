# Locality validation

The locality module retains its historical angle names: beta_angle is COST and gamma_angle is MIXER, the reverse of the QAOA class. Angles are chronological and Heisenberg conjugation traverses layers backwards. Random and dense helpers now agree in sign and layer order. The random helper accepts seed and caps the number of terms by the number of available locality strings. Qiskit binary lists use the rightmost tensor factor for qubit zero. XI counting retains unrounded complex coefficients, applies the requested tolerance directly, and returns count zero for the empty expansion. Its maximum and average locality are then zero by convention.

Notebook copies of the counter now call the tested module; dependent outputs were cleared. The old elementwise-exponential helper in QAOA_locality(ver1).ipynb also delegates to the corrected implementation. Saved optimization successes give upper bounds on attainable depth for those instances, not minimum-depth theorems; unsuccessful finite optimization runs provide no lower bound. These notebooks remain exploratory outside the regression-tested paths.

Run `python -m pip install -r requirements-test.txt` then `python -m pytest tests -q`. The regression suite checks small/empty/complex coefficients, independent dense Heisenberg evolution, consistent random construction, deterministic seeds, and zero-term Hamiltonians.


The weighted graph constructor now preserves NetworkX edge weights and node insertion order. `make_H_maxCUT(G)` retains its historical dense-matrix default; `full_matrix=False` returns the diagonal. Node/qubit zero remains the rightmost Qiskit tensor factor. `make_grid(..., seed=...)` now seeds angles and sampled Hamiltonians throughout the grid.


The legacy `QAOA_class.py` now handles one-qubit mixers and both flat and column statevectors, selects optimized first-layer angles, accepts seeded optimization, and counts every restart and refinement evaluation. The batch runner `Zhores/QAOA_locality_Zhores.py` imports the validated locality kernel, performs no work at import, and seeds both angles and Hamiltonian sampling. Its positional CLI and CSV row layout are retained; coefficients are written with full floating precision instead of rounding to three decimal places. Example from the repository: `python Zhores/QAOA_locality_Zhores.py 3 2 3 2 /tmp/qaoa-locality 13`. The index maps to `seed=index%200`, as before. Legacy saved batch files predate these corrections.
