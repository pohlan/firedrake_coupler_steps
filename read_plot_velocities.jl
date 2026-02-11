using Glob, Dates, ProgressMeter, Statistics, CSV, DataFrames, Plots, NCDatasets
import GeoArrays

# some plotting stuff
Plots.scalefontsizes()     # resets font sizes
Plots.scalefontsizes(2.7)
wsize = (1500, 900)
points_palette = :seaborn_colorblind6

# paths
data_path = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
fls = readdir(data_path, join=true)

# read in coordinates
# crds = CSV.read("/home/annegret/Projects/coupled_modeling/GrisVels/points_to_plot.csv", DataFrame)

crds = DataFrame("X" => [-223885,-216304,-215217.697,-210017],
                 "Y" => [-2.49326e6,-2.50337e6,-2510443.872,-2.52700e6])

function get_dates_from_fname(fl)
    sp = split(fl, "_")
    yr = parse(Int, "20"*sp[end-3][end-1:end])
    mm = Dates.LOCALES["english"].month_abbr_value[sp[end-3][3:5]]
    dd = parse(Int, sp[end-3][1:2])
    return Date(yr,mm,dd)
end

# loop through files
dys = Array{Date, 1}(undef, length(fls))
vs = zeros(Float32, size(crds,1), length(fls))
@showprogress for (i,fl) in enumerate(fls)
    d = get_dates_from_fname(fl)
    dys[i] = d
    ga    = GeoArrays.read(fl)
    ics   = [GeoArrays.indices(ga, (xi,yi)).I for (xi,yi) in eachrow(crds)]    # crds.X and .Y need to be in Float for this to work

    for (j,(cx,cy)) in enumerate(ics)
        # mat     = collect(skipmissing(ga.A[cx-2:cx+2, cy-2:cy+2,1]))
        # vs[j,i] = mean(mat)
        mat     = ga.A[cx, cy,1]
        vs[j,i] = ismissing(mat) ? NaN : mat  # initiate array with Union{Float, missing} instead
    end
end
# sort after dates
p = sortperm(dys)
sort!(dys)

#############################
# Plot velocity time series #
#############################

n1_plot = 1
v_plot = vs[:,p]
xlm = (Date(2017,1,1), Date(2021,2,1))

for ic in axes(v_plot,1)
    pl = Plots.plot(ylabel="Surface speed [m/yr]", leftmargin=10Plots.mm, rightmargin=25Plots.mm, bottommargin=10Plots.mm; palette=points_palette, wsize=(1200,800))
    plot!(dys[n1_plot:end],v_plot[ic,n1_plot:end], marker=:circle, markersize=8; lw=5, label="", markerstrokewidth=0, xlims=xlm, color=palette(points_palette)[ic])
    vline!(Date.(2018:2021), ls=:dash, color=:black, label="", lw=1)
    savefig("point_$(ic).png")
end

# pl = Plots.plot(ylabel="surface speed [m/yr]", leftmargin=10Plots.mm, rightmargin=10Plots.mm, bottommargin=10Plots.mm; palette=points_palette, wsize=(1200,800))
# for ic in axes(v_plot,1)
#     Plots.plot!(dys[n1_plot:end],v_plot[ic,n1_plot:end], marker=:circle, markersize=5; lw=4, label="", markerstrokewidth=0)
# end
# Plots.plot(pl)
# Plots.savefig("figures/velocities.png")

# plot an individual point from csv
ic = 4
run_index = 65
gl_idx  = Dict(1 => "", 2=> "", 3=> "3", 4=> "3")
map_idx = Dict(1 => 0, 2 => 3, 4=>1, 3=>5)
df_pt = CSV.read("parameter_runs/run_$(run_index)/$(run_index)_gl$(gl_idx[ic])_$(map_idx[ic]).csv", DataFrame)
pl = Plots.plot(ylabel="Surface speed [m/yr]", leftmargin=10Plots.mm, rightmargin=25Plots.mm, bottommargin=10Plots.mm; palette=points_palette, wsize=(1200,800), legendfontsize=15, tickfontsize=16, guidefontsize=21)
plot!(df_pt.time, df_pt.U_model, color=:black, lw=4, label="model", ls=:dot)
plot!(dys[n1_plot:end],v_plot[ic,n1_plot:end], marker=:circle, markersize=5; lw=4, markerstrokewidth=0, xlims=xlm, color=palette(points_palette)[ic], label="observations")
vline!(Date.(2017:2020), ls=:dash, color=:black, label="", lw=1)

mode = Dict(1 => "spring_speedup", 2 => "winter_speedup", 4=> "fall_min", 3=>"steep_winter")
savefig("parameter_runs/plots/run_$(run_index)_$(mode[ic]).png")



############
# Plot map #
############

# read, determine bounding box
ga = GeoArrays.read("/home/annegret/Projects/coupled_modeling/GrisVels/GL_vel_mosaic_Monthly_01Sep22_30Sep22_vv_v05.0_clipped.tif")
x0, xend = 2100,2420
y0, yend = 9250,9450
ga_sub = ga[x0:xend, y0:yend]

colors = palette(points_palette)[1:size(crds,1)]
p_map = Plots.plot(ga_sub, alpha=0.9, xlabel="Easting [m]", ylabel="Northing [m]", colorbar_title="surface speed [m/yr]", cmap=:thermal, leftmargin=10Plots.mm, rightmargin=10Plots.mm; wsize, grid=false)
for ((xi,yi),col) in zip(eachrow(crds), colors)
    Plots.scatter!([xi], [yi]; label="", marker=:circle, markersize=15, color=col, markerstrokewidth=1)
end
Plots.plot(p_map)
savefig("map_slides.png")

##############
# plot inset #
##############
# https://discourse.julialang.org/t/how-to-draw-a-rectangular-region-with-plots-jl/1719
# https://discourse.julialang.org/t/plots-jl-with-inset/102936/10 at the bottom

# read in a map of all of Greenland
ds = NCDataset("/home/annegret/Projects/svd_IceSheetDEM/data/bedmachine/bedmachine_g1200.nc")
gris = ds["surface"][:,:]
insert_plt = ones(size(gris))
insert_plt[ismissing.(gris) .|| gris .== 0] .= NaN

# get coordinates of bounding box
pts = [GeoArrays.coords(ga, (xi,yi)) for (xi,yi) in zip([x0,xend], [y0,yend])]
pxs = [pi[1] for pi in pts]
pys = [pi[2] for pi in pts]
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

# plot as insert to previous plot p_map
ins = bbox(0.72,0.15,0.14,0.24, :left)
plot!(p_map, inset=ins, subplot=2)
heatmap!(p_map[2], ds["x"][:], ds["y"][:], insert_plt', cmap=:Greys, fillalpha=0.2, grid=false, label="", cbar=false, axis=([],false))
plot!(p_map[2], rectangle(diff(pxs)[1],diff(pys)[1],pxs[1],pys[1]), fillalpha=0, linewidth=2, label="")
plot!(p_map[2], [pxs[2]+4e5,pxs[2]+1e5], [pys[1]+6e5,pys[1]+1e5], arrow=true, color=:black, lw=2, label="")

savefig("figures/map.png")


# plot(p_map, pl)





ga = GeoArrays.read("data/ITS_LIVE/GRE_G0240_0000_v.tif", masked=false)
x, y = GeoArrays.ranges(ga)

B = ga.A[:,:,1]
B[B .== -32767.0] .= NaN

heatmap(x, sort(y), B[:,end:-1:1]', cmap=:dense, clims=(0,800), grid=false, aspect_ratio=1, axis=([],false), wsize=(1000, 1600))


# twilight