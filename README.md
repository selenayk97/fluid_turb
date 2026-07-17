# fluid_turb 2026 grad course
## Simulating Kelvin-Helmholtz Instabilities in a Stratified Shear Layer Project ##


Code is a 2D incompressible Boussinesq pseudo-spectral solver that uses the 2D incompressible Boussinesq equations in spectral
form with Runge Kutta (RK4) time stepping, varying Ri and Re
while tracking KH billow development and mixing diagnostics
#
Features:
   - stratified tanh shear layer
   - controllable Richardson number Ri
   - controllable Reynolds number Re
   - perturbation amplitude / phase controls
   - RK4 time stepping
   - diagnostics: TKE, enstrophy, mixing thickness, growth
   - snapshot, figure and file saving to folder
#
Dependencies: numpy, matplotlib, quiver, 
#
Notes:
   - 2D model
   - useful for KH growth, billow roll-up, nonlinear breakdown,
     mixing trends, and perturbation sensitivity

Governing Equations
=============================================

This code solves the 2D incompressible Boussinesq equations in
vorticity-streamfunction form.

Coordinates:

    x = horizontal coordinate
    y = vertical coordinate

Velocity:

    u = horizontal velocity
    w = vertical velocity

Streamfunction definition:

    u = ∂ψ/∂y
    w = -∂ψ/∂x

Vorticity:

    ω = ∂w/∂x - ∂u/∂y

Poisson equation:

    ∇²ψ = -ω

Governing equations:

    ∂ω/∂t + J(ψ, ω) = ∂b/∂x + ν∇²ω

    ∂b/∂t + J(ψ, b) = κ∇²b

where:

    b   = buoyancy
    ν   = kinematic viscosity
    κ   = scalar diffusivity
    
    J(ψ,q) = (∂ψ/∂x)(∂q/∂y) - (∂ψ/∂y)(∂q/∂x)

Base state:

    U(y) = U0 tanh((y-yc)/h)

    b0(y) = B0 tanh((y-yc)/δ)

Control parameters:

    Reynolds number:
        Re = U0 h / ν

    Richardson number:
        Ri ~ B0 h / U0²

Diagnostics computed:

    - Turbulent kinetic energy (TKE)
    - Perturbation kinetic energy (PKE)
    - Enstrophy
    - Growth factor = PKE(t) / PKE(0)
    - Mixing layer thickness
    - Dominant wavelength (unmark to run)

# Workflow in one clean path:

1. configure one run  
2. run the simulation  
3. save a consistent `.npz` output  
4. plot time histories  
5. plot Reynolds-stress summaries  
6. plot kinetic-energy spectra from saved snapshots
