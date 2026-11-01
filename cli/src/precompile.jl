# =================================================================================================
# The precompile workload: every call the command line makes, run once while the package is
# built, so that a command starts compiled (each `equip` call is a new Julia process; without
# this, `equip fit T2Decay t2.nii.gz` spent 17 s compiling, against 0.8 s to load). It writes a
# small dataset for each built-in model (2×2×1 voxels, noise-free from `_TRUTHS`) and fits it
# through `main`, exactly as the command does: reading NIfTI files and a BIDS sidecar, the flags,
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
    QMTSIR = ["--protocol", "SIR_5POINT"])
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
    QMTSIR = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1.0))

@setup_workload begin
    dir = mktempdir()
    quiet = IOBuffer()
    @compile_workload begin
        for (name, M) in pairs(MODELS)
            flags = _EXAMPLES[name]
            m = buildmodel(M, parseargs(flags), (;); warn = quiet)
            s = signal(m, merge(defaults(m), _TRUTHS[name]))
            data = Float32.(reshape(repeat(s', 4), 2, 2, 1, :) .* reshape([1.0, 0.9, 1.1, 0.95], 2, 2, 1, 1))
            f = joinpath(dir, "$name.nii.gz")
            niwrite(f, NIVolume(data))
            main(["fit", String(name), f, flags..., "--out", joinpath(dir, "$(name)_")]; io = quiet, err = quiet)
        end
        # A sidecar's EchoTime, a mask, the statistics' maps, --save and --config, --help, a mistake
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        f = joinpath(dir, "t2.nii.gz")
        niwrite(f, NIVolume(Float32.(reshape(repeat(signal(m, _TRUTHS.T2Decay)', 4), 2, 2, 1, :))))
        open(io -> JSON.json(io, Dict("EchoTime" => te)), joinpath(dir, "t2.json"), "w")
        niwrite(joinpath(dir, "mask.nii.gz"), NIVolume(UInt8[1 1; 1 0;;;]))
        cnf = joinpath(dir, "t2.cnf")
        main(["fit", "T2Decay", f, "--mask", joinpath(dir, "mask.nii.gz"), "--stderror",
              "--save", cnf, "--out", joinpath(dir, "t2_")]; io = quiet, err = quiet)
        main(["fit", "--config", cnf, "--T2", "free,start=0.05"]; io = quiet, err = quiet)
        main(["fit", "QMTSIR", "--help"]; io = quiet, err = quiet)
        main(["fit", "T2Decay", f, "--nonsense", "1"]; io = quiet, err = quiet)
        main(["--help"]; io = quiet, err = quiet)
    end
end
