# The timings (out/timings.tsv, from vf.jl and eq.jl "time") as a Markdown table
include(joinpath(@__DIR__, "cases.jl"))
rows = [split(l, '\t') for l in readlines(joinpath(@__DIR__, "out", "timings.tsv"))]
t(pkg, c, nt, prec, mode) = parse(Float64, only(r[6] for r in rows if r[1] == pkg && r[2] == c && r[3] == string(nt) && r[4] == prec && r[5] == mode))
f(x) = string(round(x; sigdigits = 3))
println("| model | own starts, 1 thread: VoxelFit / EQUIP (ratio) | own starts, 4 threads | same starts, 1 thread | same starts, 4 threads | own starts, Float32, 4 threads |")
println("|---|---|---|---|---|---|")
for c in CASES
    cell(nt, prec, mode) = (v = t("VoxelFit", c.name, nt, prec, mode); e = t("EQUIP", c.name, nt, prec, mode); "$(f(v)) / $(f(e)) ($(round(e / v; digits = 2))×)")
    println("| $(c.name) | ", cell(1, "Float64", "default"), " | ", cell(4, "Float64", "default"), " | ", cell(1, "Float64", "given"), " | ",
            cell(4, "Float64", "given"), " | ", cell(4, "Float32", "default"), " |")
end
