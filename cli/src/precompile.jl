# =================================================================================================
# The precompile workload: every call the command line makes, run once while the package is
# built, so that a command starts compiled (each `equip` call is a new Julia process; without
# this, `equip fit T2Decay t2.nii.gz` spent 17 s compiling, against 0.8 s to load). It writes a
# small dataset for each built-in model (2×2×1 voxels, noise-free from `_TRUTHS`) and fits it
# through `equip`, exactly as the command does: reading NIfTI files and a BIDS sidecar, the flags,
# a mask, writing the maps and their sidecars, `--save`, `--help`, and a mistake's message.
# =================================================================================================

# An example acquisition for each model, as flags: what the workload fits with.
const _EXAMPLES = (
    T1IR = ["--TI", "0.05,0.1,0.2,0.4,0.8,1.6,3.2"], R1IR = ["--TI", "0.05,0.1,0.2,0.4,0.8,1.6,3.2"],
    T1IRTD = ["--TI", "0.05,0.1,0.4,1.0,2.0", "--TD", "2.0"], R1IRTD = ["--TI", "0.05,0.1,0.4,1.0,2.0", "--TD", "2.0"],
    T1SR = ["--TD", "0.1,0.3,0.8,1.5,3.0"], R1SR = ["--TD", "0.1,0.3,0.8,1.5,3.0"],
    T1MFA = ["--FA", "2,5,10,15,20", "--TR", "0.015"], R1MFA = ["--FA", "2,5,10,15,20", "--TR", "0.015"],
    T2Decay = ["--TE", "0.01:0.01:0.32"], R2Decay = ["--TE", "0.01:0.01:0.32"],
    T2DecayOffset = ["--TE", "0.01:0.01:0.32"], R2DecayOffset = ["--TE", "0.01:0.01:0.32"],
    T2BiexpDecay = ["--TE", "0.01:0.01:0.32"], R2BiexpDecay = ["--TE", "0.01:0.01:0.32"],
    ADCDecay = ["--b", "0,500,1000,1500,2000,2500"], ADCKurtosis = ["--b", "0,500,1000,1500,2000,2500"],
    IVIM = ["--b", "0,10,20,50,100,200,400,600,800"],
    QMTSIR = ["--protocol", "SIR_5POINT"], QMTSIREqualR1 = ["--protocol", "SIR_5POINT"])
# True values for the workload's data (the fixed parameters take the model's defaults); typical
# white-matter-like values.
const _TRUTHS = (
    T1IR = (M0 = 1.0, Sf = -0.95, T1 = 1.0), R1IR = (M0 = 1.0, Sf = -0.95, R1 = 1.0),
    T1IRTD = (M0 = 1.0, Sf = -0.95, T1 = 1.0), R1IRTD = (M0 = 1.0, Sf = -0.95, R1 = 1.0),
    T1SR = (M0 = 1.0, T1 = 1.0), R1SR = (M0 = 1.0, R1 = 1.0),
    T1MFA = (M0 = 1.0, T1 = 1.0), R1MFA = (M0 = 1.0, R1 = 1.0),
    T2Decay = (S0 = 1.0, T2 = 0.08), R2Decay = (S0 = 1.0, R2 = 12.5),
    T2DecayOffset = (S0 = 1.0, T2 = 0.08, C = 0.0), R2DecayOffset = (S0 = 1.0, R2 = 12.5, C = 0.0),
    T2BiexpDecay = (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08),
    R2BiexpDecay = (S0 = 1.0, fshort = 0.2, R2short = 50.0, R2long = 12.5),
    ADCDecay = (S0 = 1.0, ADC = 8e-4), ADCKurtosis = (S0 = 1.0, ADC = 1e-3, K = 1.0),
    IVIM = (S0 = 1.0, f = 0.1, Dstar = 0.02, D = 1e-3),
    QMTSIR = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1.0),
    QMTSIREqualR1 = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1.0))

# DTI and DKI's example: a b = 0 image and two shells of 15 directions (31 images, DKI's 22
# parameters need two shells); white-matter-like D (eigenvalues 1.7, 0.4, 0.3 × 10⁻³ mm²/s along
# x, y, z) and, for DKI, an isotropic kurtosis K = 1 (W in V = MD²·W form, as DKI fits it).
const _DWI_B = [0.0; fill(1000.0, 15); fill(2000.0, 15)]
const _DTI_TRUTH = (S0 = 1.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0)
const _DKI_TRUTH = let v = (0.8e-3)^2
    merge(_DTI_TRUTH, (V1111 = v, V2222 = v, V3333 = v, V1112 = 0.0, V1113 = 0.0, V1222 = 0.0,
                       V2223 = 0.0, V1333 = 0.0, V2333 = 0.0, V1122 = v / 3, V1133 = v / 3,
                       V2233 = v / 3, V1123 = 0.0, V1223 = 0.0, V1233 = 0.0))
end
# Write a 2×2×1 diffusion dataset for model M at `truth`, with its .bval and .bvec beside it.
function _writedwi(f, M, truth)
    g = directionset(15)
    G = reduce(hcat, [[[0.0, 0.0, 0.0]]; collect.(g); collect.(g)])     # 3 × 31, as a .bvec
    m = M(; diffusionencoding(_DWI_B, G)...)
    s = signal(m, merge(defaults(m), truth))
    writeimage(f, Float32.(reshape(repeat(s', 4), 2, 2, 1, :) .* reshape([1.0, 0.9, 1.1, 0.95], 2, 2, 1, 1)))
    stem = replace(f, r"\.nii(\.gz)?$" => "")
    write(stem * ".bval", join(_DWI_B, ' ') * "\n")
    write(stem * ".bvec", join((join(G[i, :], ' ') for i in 1:3), "\n") * "\n")
    return f
end

@setup_workload begin
    dir = mktempdir()
    quiet = IOBuffer()
    @compile_workload begin
        for (name, M) in pairs(MODELS)
            haskey(_EXAMPLES, name) || continue          # DTI and DKI: from files, below
            flags = _EXAMPLES[name]
            m = buildmodel(M, parseargs(flags), Dict{Symbol,Vector{Float64}}(); warn = quiet)
            s = signal(m, merge(defaults(m), _TRUTHS[name]))
            data = Float32.(reshape(repeat(s', 4), 2, 2, 1, :) .* reshape([1.0, 0.9, 1.1, 0.95], 2, 2, 1, 1))
            f = joinpath(dir, "$name.nii.gz")
            writeimage(f, data)
            equip(["fit", String(name), f, flags..., "--out", joinpath(dir, "$(name)_")]; io = quiet, err = quiet)
        end
        # DTI and DKI, the way they are called: the gradient directions from the FSL .bval and
        # .bvec beside the data (dcm2niix's), and their derived maps (MD, FA, ..., MK, ...)
        dwi = joinpath(dir, "dwi.nii.gz")
        for (M, truth) in ((DTI, _DTI_TRUTH), (DKI, _DKI_TRUTH),
                           (StickZeppelin, (S0 = 1.0, vax = 0.6, Dax = 1.9e-3, theta = π / 2, phi = 0.0)),
                           (BallStick, (S0 = 1.0, f = 0.5, d = 1.2e-3, theta = π / 2, phi = 0.0)),
                           (NODDI, (S0 = 1.0, vic = 0.6, viso = 0.1, ODI = 0.2, theta = π / 2, phi = 0.0)))
            _writedwi(dwi, M, truth)
            equip(["fit", String(nameof(M)), dwi, "--out", joinpath(dir, "$(nameof(M))_")]; io = quiet, err = quiet)
        end
        equip(["fit", "DTI", "--help"]; io = quiet, err = quiet)
        # A sidecar's EchoTime, a mask, the statistics' maps, --save and --config, --help, a mistake
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        f = joinpath(dir, "t2.nii.gz")
        writeimage(f, Float32.(reshape(repeat(signal(m, _TRUTHS.T2Decay)', 4), 2, 2, 1, :)))
        writesidecar(f, Dict("EchoTime" => te))
        writeimage(joinpath(dir, "mask.nii.gz"), UInt8[1 1; 1 0;;;])
        cnf = joinpath(dir, "t2.cnf")
        equip(["fit", "T2Decay", f, "--mask", joinpath(dir, "mask.nii.gz"), "--stderror",
              "--save", cnf, "--out", joinpath(dir, "t2_")]; io = quiet, err = quiet)
        equip(["fit", "--config", cnf, "--T2", "free,start=0.05"]; io = quiet, err = quiet)
        equip(["fit", "QMTSIR", "--help"]; io = quiet, err = quiet)
        equip(["info", f]; io = quiet, err = quiet)
        equip(["info", f, joinpath(dir, "mask.nii.gz"), "--find", "echo"]; io = quiet, err = quiet)
        # crlb, montecarlo and design: the SIR schemes and T2 decay (each model's crlb compiles
        # its second derivatives, so the others compile at their first use, a few seconds)
        tsv, cnf2 = joinpath(dir, "t.tsv"), joinpath(dir, "d.cnf")
        sir = ["QMTSIR", "--protocol", "SIR_4POINT", "--snr", "200", "--kmf", "12.5"]
        equip(["crlb"; sir; "--tissue"; "SIR_WM_I"]; io = quiet, err = quiet)
        equip(["crlb"; sir; "--tissue"; "SIR_WM_I,SIR_lesion"; "--out"; tsv]; io = quiet, err = quiet)
        equip(["crlb", "QMTSIR", "--protocol", "SIR_5POINT", "--tissue", "SIR_WM_I,SIR_lesion",
               "--snr", "100,200", "--kmf", "free", "--cov", "biased", "--out", tsv]; io = quiet, err = quiet)
        equip(["crlb", "T2Decay", "--TE", "0.01:0.01:0.32", "--truth", "S0=1000,T2=0.08", "--sigma", "10",
               "--vary", "T2=0.04:0.04:0.12", "--out", tsv]; io = quiet, err = quiet)
        equip(["crlb", "T2Decay", "--TE", "0.01:0.01:0.32", "--truth", "S0=1000,T2=0.08", "--sigma", "10"];
              io = quiet, err = quiet)
        equip(["montecarlo"; sir; "--tissue"; "SIR_WM_I"; "--noise"; "rician"; "--n"; "20"; "--seed"; "1"];
              io = quiet, err = quiet)
        equip(["montecarlo"; sir; "--tissue"; "SIR_WM_I,SIR_lesion"; "--noise"; "rician"; "--n"; "20";
               "--seed"; "1"; "--out"; tsv]; io = quiet, err = quiet)
        equip(["montecarlo", "T2Decay", "--TE", "0.01:0.01:0.32", "--truth", "S0=1000,T2=0.08", "--sigma",
               "10", "--noise", "gaussian", "--n", "20", "--seed", "1"]; io = quiet, err = quiet)
        equip(["design"; sir; "--tissue"; "SIR_WM_I,SIR_lesion"; "--acquisition"; "TI=0.01..10,TD=0.01..10";
               "--readout"; "0.3"; "--starts"; "1"; "--generations"; "2"; "--seed"; "1"; "--out"; cnf2];
              io = quiet, err = quiet)
        equip(["crlb", "QMTSIR", "--help"]; io = quiet, err = quiet)
        equip(["montecarlo", "QMTSIR", "--protocol", "SIR_4POINT", "--tissue", "SIR_WM_I", "--snr", "200"];
              io = quiet, err = quiet)
        equip(["fit", "T2Decay", f, "--nonsense", "1"]; io = quiet, err = quiet)
        equip(["--help"]; io = quiet, err = quiet)
    end
end
# The entry point as the command calls it, writing to the real stdout and stderr: a file, a pipe
# or a terminal; and a few of Base's methods the commands reach by dynamic dispatch.
precompile(main, (Vector{String},))
precompile(_run, (Vector{String}, IO, IO))
precompile(_flush, (IOBuffer, IO, IOBuffer, IO))
for S in (IOStream, Base.PipeEndpoint, Base.TTY)
    precompile(write, (S, Vector{UInt8}))
    precompile(flush, (S,))
end
precompile(in, (String, Set{String}))
precompile(in, (Symbol, Tuple{Symbol}))
precompile(size, (Array{Float64,4},))
precompile(size, (Array{Float64,4}, Int))
precompile(size, (Array{Float32,4},))
precompile(size, (Array{Float32,4}, Int))
precompile(size, (Array{FitStatus,3},))
precompile(getindex, (Dict{String,Any}, String))
