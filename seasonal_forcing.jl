using Plots

day   = 3600*24
year  = day*365 #      31536000
# lr    = -0.0075
lr    = -0.005
DDF   = 0.01/day
basal = 7.93e-11

shmpF_DT = Dict("F1" => -6,
                "F2" => -3,
                "F3" =>  0,
                "F4" =>  3,
                "F5" =>  6)
shmpA_DT = Dict("D1" => -4,
                "D2" => -2,
                "D3" =>  0,
                "D4" =>  2,
                "D5" =>  4)



# surface(x,y) = 100(x+200)^(1/4) + 1/60*x - 2e10^(1/4) + 1  # valley
# xs = 0:100:6e3
# ys = 0
# z_s = surface.(xs, ys)

surface(x,y) = 6*( sqrt(x+5e3) - sqrt(5e3) ) + 1           # sqrt
xs = 0:100:100e3
ys = 0
z_s = surface.(xs, ys)

temp(t, DT) = -16*cos(2*pi/year*t)- 5 + DT
runoff(z_s,t, DT) = max(0, (z_s*lr+temp.(t, DT))*DDF) + basal

t = 1:day:year


# F
p = plot()
for (suite, DT) in shmpF_DT
    m = runoff.(minimum(z_s),t, DT)
    plot!(t/day, m, label=suite)
end
plot(p)
savefig("DT_spatialmax_F.png")

# A
p = plot()
for (suite, DT) in shmpA_DT
    # m = runoff.(minimum(z_s),t, DT)
    # m = max.(390*lr.+temp.(t, DT),0)
    m = -10*cos.(2*pi/year*t).- 2
    plot!(t/day, m, label=suite)
end
plot(p)
savefig("DT_spatialmax_A.png")



