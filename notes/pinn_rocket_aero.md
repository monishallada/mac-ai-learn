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

## Extended pseudocode sketch (PyTorch-ish)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256, depth=5):
        super().__init__()
        self.input = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
            ) for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(depth)])

    def forward(self, x):
        h = F.relu(self.input(x))
        for block, norm in zip(self.blocks, self.norms):
            h = F.relu(norm(h + block(h)))  # residual path for stability
        return h

class AeroPINN(nn.Module):
    def __init__(self, d_in, λ_phys=1.0):
        super().__init__()
        self.backbone = ResidualMLP(d_in)
        self.aero_head = nn.Linear(256, 6)       # CL, CD, CY, Cl, Cm, Cn
        self.stab_head = nn.Linear(256, 3)       # static margin, dutch/directional damping
        self.λ_phys = λ_phys

    def forward(self, x):
        h = self.backbone(x)
        aero = self.aero_head(h)
        stability = self.stab_head(h)
        return aero, stability

def lift_relation_residual(aero, state):
    CL, CD, CY, Cl, Cm, Cn = aero.T
    rho = state["rho"]; V = state["V"]; S = state["S"]; q = 0.5 * rho * V**2
    L = q * S * CL
    # Target lift for trimmed climb, e.g., mg * cos(bank)
    target_L = state["mass"] * state["g"] * torch.cos(state["bank"])
    return (L - target_L)**2

def moment_balance_residual(aero, state):
    _, _, _, Cl, Cm, Cn = aero.T
    lever = state["lever_arm"]
    # Enforce near-zero pitching moment for trimmed flight; yaw moment countering crosswind
    pitch_res = (Cm + lever * Cl)**2
    yaw_target = -state["crosswind_force"] * state["fin_span"]  # weathercocking tendency
    yaw_res = (Cn - yaw_target)**2
    return pitch_res + yaw_res

def monotonicity_residual(aero_fn, base_state, fin_scale=1.05):
    # Encourage larger fins -> stronger directional stability (more negative Cn derivative)
    x_base = base_state["x"]
    x_scaled = x_base.clone(); x_scaled[:, base_state["fin_idx"]] *= fin_scale
    with torch.enable_grad():
        x_base.requires_grad_(True)
        aero_base, _ = aero_fn(x_base)
        Cn_base = aero_base[:, 5]
        grad = torch.autograd.grad(Cn_base.sum(), x_base, create_graph=True)[0][:, base_state["fin_idx"]]
    return F.relu(grad).mean()  # penalize non-negative slope

def collocation_sampler(batch_size, ranges):
    # Randomly sample AoA, sideslip, density, wind to enforce physics over the envelope
    samples = {k: torch.empty(batch_size).uniform_(lo, hi) for k, (lo, hi) in ranges.items()}
    samples["x"] = torch.stack([samples[k] for k in ranges.keys()], dim=1)
    return samples

def loss_fn(model, batch, collocation, base_state):
    aero_pred, stab_pred = model(batch["x"])
    data_loss = F.mse_loss(aero_pred, batch["labels"])

    aero_c, _ = model(collocation["x"])
    lift_res = lift_relation_residual(aero_c, collocation).mean()
    moment_res = moment_balance_residual(aero_c, collocation).mean()
    mono_res = monotonicity_residual(model, base_state)
    physics_loss = lift_res + moment_res + mono_res

    return data_loss + model.λ_phys * physics_loss

def train_step(model, opt, batch, collocation, base_state):
    opt.zero_grad()
    loss = loss_fn(model, batch, collocation, base_state)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {"loss": loss.item(), "data": batch["labels"].shape[0]}

# Usage snippet
model = AeroPINN(d_in=12, λ_phys=5.0)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
ranges = {"aoa": (-5, 15), "beta": (-10, 10), "rho": (0.7, 1.2), "wind": (0, 12)}
for step in range(10_000):
    batch = next(data_loader)  # provides x, labels, plus metadata (S, rho, etc.)
    collocation = collocation_sampler(256, ranges)
    base_state = {"x": batch["x"].detach(), "fin_idx": 3, "fin_span": batch["fin_span"], "crosswind_force": batch["crosswind_force"], "lever_arm": batch["lever_arm"], "g": 9.81, "bank": batch["bank"]}
    metrics = train_step(model, opt, batch, collocation, base_state)
    if step % 100 == 0:
        print(step, metrics)
```

## Usage loop
1) Precompute/collect CFD or wind-tunnel coefficients; normalize inputs; set collocation sampling over winds, densities, AoA, Mach.
2) Train PINN with mixed data + physics loss; validate on withheld conditions.
3) Expose an interface that accepts new designs and atmospheric states and returns coefficients plus stability metrics to answer “how do bigger fins change stability under 10 mph crosswinds?”
