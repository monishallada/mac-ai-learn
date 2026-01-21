# PINN for Rocket Aerodynamics (concept)

Goal: build a physics-informed neural network (PINN) that maps rocket design parameters and atmospheric conditions to aerodynamic performance, while enforcing known flight physics.

## Inputs
- Geometry/design: fin span/chord/thickness, body diameter/length, nose cone shape, mass distribution, center of gravity.
- Control/surface trims: fin cant angle, roll/pitch/yaw trim settings.
- Environment: altitude, temperature, pressure, air density, wind speed/direction (e.g., 10 mph crosswind), turbulence intensity.
- Operating point: speed (Mach), angle of attack, sideslip angle.

## Outputs (examples)
- Aerodynamic coefficients: `CL`, `CD`, `CY`, `Cl`, `Cm`, `Cn`.
- Stability metrics: static margin, weathercock stability, damping ratios.
- Performance: max crosswind tolerance, expected drift, control authority margin.

## PINN structure
- Network: MLP over normalized inputs; small residual blocks help with wide operating regimes.
- Physics heads: predict aero coefficients and derived stability metrics.
- Loss
  - Data loss: MSE between predicted coefficients and CFD/tunnel/flight data.
  - Physics loss: residuals from simplified governing equations (e.g., steady-state Navier-Stokes approximations, lift/drag relations, moment balance) evaluated at collocation points.
  - Regularizers: smoothness across angle-of-attack and wind perturbations; monotonicity constraints for fin size vs stability margin.
- Training data
  - Supervised: CFD samples, past flight tests.
  - Collocation: randomly sample operating points (AoA, wind vectors, densities) to enforce physics residuals even where data is sparse.

## Example “what-if” query
- Scenario: increase fin area by 5% while holding mass and CG fixed; evaluate at 10 mph (≈4.5 m/s) crosswind, sea-level density, AoA 3°.
- PINN response pattern: higher `CY` and `Cn` magnitude → improved directional stability; slightly higher `CD`; updated static margin and weathercock tendency; drift reduced at launch but check structural loads.

## Pseudocode sketch (PyTorch)
```python
import torch

def forward(x):
    # x: [batch, features] with geometry + environment
    h = mlp(x)
    aero = aero_head(h)          # CL, CD, CY, Cl, Cm, Cn
    stability = stability_head(h)  # static margin, damping ratios
    return aero, stability

def loss_fn(batch, collocation):
    aero_pred, stab_pred = forward(batch.inputs)
    data_loss = mse(aero_pred, batch.labels)

    # Physics residuals at collocation points
    aero_c, _ = forward(collocation.inputs)
    lift_res = lift_relation_residual(aero_c, collocation)
    moment_res = moment_balance_residual(aero_c, collocation)
    physics_loss = lift_res.mean() + moment_res.mean()

    return data_loss + λ_phys * physics_loss
```

## Usage loop
1) Precompute/collect CFD or wind-tunnel coefficients; normalize inputs; set collocation sampling over winds, densities, AoA, Mach.
2) Train PINN with mixed data + physics loss; validate on withheld conditions.
3) Expose an interface that accepts new designs and atmospheric states and returns coefficients plus stability metrics to answer “how do bigger fins change stability under 10 mph crosswinds?”
