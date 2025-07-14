# firedrake_coupler_steps

### Step 1: only GlaDS, only sheets

- there were some numbers added in `q` calculation that were too high (1e-3 instead of eps) --> made `phi` orders of magnitude too small
- now producing the expected result from A1 shmip case

### Step 2: only GlaDS, sheets + channels

- using CR elements for channels
- not quite converging well yet

### Step 2b: only GlaDS, sheets + channels

- reformatting a bit, get rid of functions
- DGT elements for channels
- adjusting time stepping, sort of working now although not quite the result of GlaDS-matlab yet; comparison to GlaDS-matlab so far only based on maximum `phi` which should be around 4.0e6 for A1

### Step 3: only GlaDS, sheets + channels, continued, only A1

- interpolating GlaDS-matlab result onto element dofs and compare
- initializing with A1 steady state for higher melt
- switching to CG1 for `h`, gives better agreement with GlaDS matlab; increasing the resolution above `nx, ny = 75, 25` doesn't really improve the agreement further (GlaDS-matlab roughly 100x30 dofs)
- for larger `dt_max` or `timestep_increase_fraction`, may need to run for longer to reach steady state
- works for A1 now as is, but not necessarily for the other cases

### Step 4: same as before but with GLADS class; made it work for A1-A6

- putting stuff in different files so that main file is much shorter and concentrates on numerical stuff to tweak to make different test cases run (`m`, `dt`, `dt_max`, initial conditions etc..)
- everything else exactly the same as before
- works well for A1-A6, just need to make sure it runs for long enough to reach steady state where the solver converges in one step every time (for larger `dt` it may need to be run for longer)
- Works for A1-A6 when initialized with an arbitrary field, no need to save a previous steady state. For A6 it works better when the channel cross-section `S` is initialized being in the order of 10 and with a linear dependence on the `x` coordinate; for the others 0.001 is ok.

![Screenshot from 2025-05-12 15-56-37](https://github.com/user-attachments/assets/f1d696e9-0fde-4f0f-b991-8d832af086ac)
Left: 🔥🐉, right: GlaDS-matlab


### Step 5: implementing the E test cases (valley geometry)

- mesh now generated with `make_valley_mesh_shmip.py` using `gmsh` rather than in firedrake directly; taking the outline function y(x) from the shmip instruction website and generating points along it
- working for E1-E4 at the moment; time step has to reduce a couple times but recovers; not for E5, it cannot get above ~0.06 which takes ages go reach steady state; changing the mesh slightly or the initial conditions has not lead to success yet
- The trick for getting E5 to run was to apply a little trick copied from GlADS: `f` switches not at exactly `S=0` or `dPds=0` but at a certain threshold, and between 0 and that threshold it is a number betwen 0 and 1.
- Also, it helps to initialize E5 with E2 or E3.


### Step 6: D test cases (ice sheet geometry with seasonal melt input)

- working well for D1-D5; needed some trial and error to figure out which time steps are reasonal, here used an ad hoc linear function depending on `m`, always in the range of 5h to 15h.
- for D5, channels nicely form and close seasonally

### Step 7: F test cases (valley geometry with seasonal melt input)

- taking over setup from step 6 including time stepping; works pretty well for F1-F5
- just need to make sure that the steady state for initialization is run with the same `m` as the background level of the seasonal input (which is different than the value for all `E` cases)

### Step 8: GlaDS on real geometry (russel basin Greenland)

- run into steady state, works well with bedmachine data, no smoothing or similar of the geometry
- manually selected a outline for the mesh; Dirichlet BC on full front line although it also includes some marginal lines in between termini of different glaciers (probably change at some point??)
- constant melt water input `m` (as always runs better when this is low, not working for 1e-7 right now)
- no channels with `m=1e-8`

### Step 9a: first time running ice flow; SpecFO, no ice thickness evolution only velocity, on SHMIP E1 valley geometry

- not coupled to GlaDS for now, `N` set to constant fraction of overburden
- Currently, it is too slippery and even for `N = 1.0*rho*g*H` the velocity is highest at the front and non-zero where ice thickness is almost zero; confusing at first; `N` can be set to 1e10 to get more realistic spatial distribution of velocity, probably need to adjust some parameters
