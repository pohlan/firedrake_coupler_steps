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

### Step 4: same as before but with GLADS class, different structure; made it work for A1-A6

- putting stuff in different files so that main file is much shorter and concentrates on numerical stuff to tweak to make different test cases run (`m`, `dt`, `dt_max`, initial conditions etc..)
- everything else exactly the same as before
- works well for A1-A6, just need to make sure it runs for long enough to reach steady state where the solver converges in one step every time (for larger `dt` it may need to be run for longer)
- Works for A1-A6 when initialized with an arbitrary field, no need to save a previous steady state. For A6 it works better when the channel cross-section `S` is initialized being in the order of 10, for the others 0.001 is ok. Randomizing the `S` initialization helps for the channels to form in the middle and not just on the lateral margins of the glacier/ice sheet.
