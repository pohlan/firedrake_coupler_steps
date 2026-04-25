using Glob, GeoArrays, Dates, Plots
# using FileIO   # optional but useful for geotiff write (if not with GeoArrays)
# using Colors

# input folder, mask year >= 2017
data_path = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/"
fls = readdir(data_path, join=true)

# helper: parse year from filename
function get_year_from_filenm(fl)
    sp = split(fl, "_")
    yr = parse(Int, "20"*sp[end-3][end-1:end])
    mm = Dates.LOCALES["english"].month_abbr_value[sp[end-3][3:5]]
    dd = parse(Int, sp[end-3][1:2])
    return yr # Date(yr,mm,dd)
end

get_year_from_filenm(fls[1])

# select files year >= 2017
files_2017plus = filter(f -> begin
        yr = get_year_from_filenm(f)
        yr !== missing && yr >= 2017
    end, files)

@info "Files since 2017:" length(files_2017plus)

# bounding box in target CRS
x0, x1 = -2.35e5, -1.75e5
y0, y1 = -2.55e6, -2.488e6

# first file for geometry
ga0 = GeoArrays.read(files_2017plus[1])
ga0_crop = GeoArrays.crop(ga0, (;min_x=x0, min_y=y0, max_x=x1, max_y=y1))
h, w = size(ga0_crop)
# transform = ga0_crop.f.geoTransform
# crs = ga0_crop.f.crs

valid_count = zeros(Int, h, w)
# iterate and count not-NaN
for (i, fn) in enumerate(files_2017plus)
    @info "reading [$i / $(length(files_2017plus))]: $fn"
    ga = GeoArrays.read(fn)
    ga = GeoArrays.crop(ga,(;min_x=x0, min_y=y0, max_x=x1, max_y=y1))
    data = ga.A[:,:,1]
    valid_count .+= .!ismissing.(data) .& .!isnan.(data)
end

# write result as GeoTIFF map
# out_geotiff = "/home/annegret/Projects/coupled_modeling/GrisVels/data/MEaSUREs/monthly/raw/valid_counts_2017plus.tif"
# GeoArrays.write(GeoArray(valid_count, geotransform=transform, crs=crs), out_geotiff)
# @info "Wrote valid-count GeoTIFF: $out_geotiff"

# plot and save PNG
heatmap(
    reverse(valid_count, dims=1),   # flip if needed display coords y upward
    color=cgrad(:viridis),
    aspect_ratio=:equal,
    title="Valid observation count per pixel since 2017",
    xlabel="x index", ylabel="y index"
)
savefig("valid_count_map_2017plus.png")
@info "Saved map PNG"
