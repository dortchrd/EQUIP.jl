function [maps, info] = equip_fit(model, data, varargin)
%EQUIP_FIT  Fit a model voxel by voxel with EQUIP, from MATLAB.
%
%   maps = equip_fit(MODEL, DATA, Name, Value, ...)
%   [maps, info] = equip_fit(...)
%
%   MODEL  the model's name: 'QMTSIR', 'T2Decay', 'T1IR', 'ADCDecay', 'DTI', 'DKI', ...
%          (all of them, and each one's options: equip('fit --help'), equip('fit QMTSIR --help'))
%   DATA   a NIfTI file, 'SIR.nii' or 'SIR.nii.gz': a 4-D image, one volume per observation;
%          several files, one per observation, as a cell array; or a MATLAB array, 4-D
%          [x y z n] or 2-D [voxels n], the last dimension the observations.
%
%   Name, Value pairs are equip fit's options, named without the dashes:
%     The acquisition, one value per volume, in the order of the volumes, TIMES IN SECONDS:
%       'TI', [1.007 0.010 0.278 0.010], 'TD', [0.010 4.174 2.730 0.684]
%     A parameter of the model:
%       fixed at a number          'kmf', 12.5
%       estimated                  'kmf', 'free'   (with a start or bounds: 'free,start=10,lower=0')
%       fixed to a map             'B1', 'B1map.nii' (or a MATLAB array, the data's size [x y z])
%     'mask'        voxels to fit: a NIfTI file, or a MATLAB array (nonzero = fit)
%     'protocol'    a named protocol, e.g. 'SIR_5POINT' (its TI and TD)
%     'stderror'    true: standard-error maps too (maps.PSR_SE, ...)
%     'likelihood'  'rician', with 'sigma' (the noise SD; a number or a map): magnitude data at
%                   low SNR fitted with the Rician noise model
%     and any other option of equip fit (equip('fit QMTSIR --help') lists them).
%   And two of this function:
%     'out'         where the map files go, as a prefix: 'results/sub01_' writes
%                   results/sub01_PSR.nii, ... Default for a NIfTI file: beside it, its name
%                   and '_' (SIR.nii: SIR_PSR.nii, ...). For a MATLAB array: a temporary folder,
%                   deleted afterwards (give 'out' to keep the files). The NIfTI maps are
%                   uncompressed; 'compress', true gzips them (.nii.gz: smaller, slower to write)
%     'verbose'     true (default): print EQUIP's summary of the fit, and every parameter as
%                   fitted (those not given in the call at EQUIP's defaults for the model)
%
%   maps   a struct with one field per map, each the data's spatial size: the fitted parameters
%          (QMTSIR: maps.PSR, maps.R1f, maps.Sf, maps.M0f), maps.status (0 not fitted: outside
%          the mask; 1-3 converged; 4 out of iterations; 5 failed; 6 invalid data) and maps.SSE.
%          NaN where not fitted. Single precision.
%   info   what was done:
%          info.parameters, every parameter of the model as the fit used it, one element each
%          (struct2table(info.parameters) shows them as a table): name; role ('free', 'fixed'
%          or 'tied'); value (a fixed parameter's value, a tied one's partner, 'a map' for a
%          map); start, lower, upper (a free parameter's; Inf where unbounded); given (true if
%          set in this call, false if at EQUIP's default for the model: check those);
%          info.command (the command run), info.output (EQUIP's output),
%          info.prefix, info.mapsfile (the maps as one .mat file; with the NIfTI maps beside it
%          for a NIfTI input), and info.sidecar, the maps' JSON record as a struct: the model,
%          the acquisition (info.sidecar.Acquisition.TI), every parameter as the fit used it,
%          the defaults too (info.sidecar.Parameters.Sm: Role 'fixed', Value 0.83;
%          .Parameters.PSR: Role 'free', Start, Lower, Upper, [] for no bound), the data and
%          mask files, the command and EQUIP's version
%
%   Example: the 4-point SIR, kmf fixed at 12.5 1/s (Dortch et al., Magn Reson Med 2018):
%     maps = equip_fit('QMTSIR', 'SIR.nii', 'mask', 'MASK.nii', ...
%                      'TI', [1007 10 278 10] / 1000, 'TD', [10 4174 2730 684] / 1000, ...
%                      'kmf', 12.5);
%     imagesc(maps.PSR(:, :, 10)); axis image; colorbar
%
%   Uses EQUIP's command line (EQUIP.jl/cli) through Julia; once after installing or updating
%   EQUIP, run equip_setup. Instructions: EQUIP.jl/matlab/README.md.
%
%   See also EQUIP_SETUP, EQUIP.

if nargin < 2
    error('equip:fit', 'give the model and the data: equip_fit(''QMTSIR'', ''SIR.nii'', ...)');
end
model = textarg(model, 'the model');
if mod(numel(varargin), 2) ~= 0
    error('equip:fit', 'options come in pairs, a name and its value (e.g. ''kmf'', 12.5)');
end

% The data: files, or an array handed over in a temporary .mat file (as variable equip_data)
vars = struct();
arraydata = isnumeric(data) || islogical(data);
if arraydata
    if ndims(data) < 2 || size(data, ndims(data)) < 2
        error('equip:fit', 'the data array: [x y z n] or [voxels n], n observations');
    end
    vars.equip_data = data;
    files = {};
else
    if iscell(data)
        files = cellfun(@(f) textarg(f, 'a data file'), data, 'UniformOutput', false);
    else
        files = {textarg(data, 'the data file')};
    end
    for k = 1:numel(files)
        f = regexprep(files{k}, '(\.mat):[A-Za-z]\w*$', '$1', 'ignorecase');   % data.mat:S
        if ~exist(f, 'file')
            error('equip:fit', 'no such file: %s (current folder: %s)', f, pwd);
        end
    end
end

% The options, as equip fit's flags; arrays (maps, a mask) handed over in the .mat file too
out = '';
verbose = true;
like = false;
flags = {};
for k = 1:2:numel(varargin)
    name = regexprep(textarg(varargin{k}, 'an option''s name'), '^-+', '');
    v = varargin{k + 1};
    switch lower(name)
        case 'out'
            out = textarg(v, '''out''');
            continue
        case 'verbose'
            verbose = logical(v);
            continue
        case 'format'
            error('equip:fit', ['equip_fit chooses the files'' format itself (the maps come ' ...
                  'back as a struct, and as NIfTI files for NIfTI data)']);
        case 'like'
            like = true;
    end
    if (ischar(v) || isstring(v))
        flags = [flags, {['--' name], char(v)}];
    elseif islogical(v) && isscalar(v)
        if any(strcmpi(name, {'stderror', 'compress'}))         % equip fit's flags without a value
            if v, flags = [flags, {['--' lower(name)]}]; end
        else
            flags = [flags, {['--' name], lower(mat2str(v))}];
        end
    elseif isnumeric(v) || islogical(v)
        if isempty(v), error('equip:fit', '''%s'' is empty', name); end
        ismap = strcmpi(name, 'mask') || (strcmpi(name, 'sigma') && numel(v) > 1) || ...
                sum(size(v) > 1) >= 2;
        if ismap
            vars.(name) = v;
            flags = [flags, {['--' name], ['@MAT:' name]}];      % the file's name, below
        else
            checkseconds(name, v);
            flags = [flags, {['--' name], numlist(v)}];
        end
    else
        error('equip:fit', '''%s'': a number, numbers, text or a logical, not a %s', name, class(v));
    end
end

% Where the arrays and the maps go
tmp = '';
if arraydata || ~isempty(fieldnames(vars))
    tmp = tempname;
    mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU> removed however this function ends
    matfile = fullfile(tmp, 'equip_inputs.mat');
    saveinputs(matfile, vars);
    flags = strrep(flags, '@MAT:', [matfile ':']);
end
if arraydata
    files = {[matfile ':equip_data']};
end
if isempty(out)
    if arraydata
        out = fullfile(tmp, 'equip_');
    else
        out = [stem(files{1}) '_'];
    end
end
ismat = ~isempty(regexpi(files{1}, '\.mat(:[A-Za-z]\w*)?$', 'once'));
format = 'both';                     % NIfTI maps to view, one .mat to load here
if ismat && ~like, format = 'mat'; end

words = [{'fit', model}, files, flags, {'--out', out, '--format', format}];
[status, txt, cmd] = equip_run(words);
txt = strtrim(txt);
if status ~= 0
    error('equip:fit', 'equip fit stopped:\n%s\n\n(the command: %s)', txt, cmd);
end
mapsfile = [out 'maps.mat'];
maps = load(mapsfile);
sidecar = jsondecode(fileread([out 'maps.json']));          % (R2016b; Octave 7)
params = parameterlist(sidecar);
if verbose
    fprintf('%s\n\n', txt);
    printparameters(params, sidecar.Model);
end
if arraydata && startsWith(out, tmp), mapsfile = ''; end   % deleted with the temporary folder
info = struct('command', cmd, 'output', txt, 'prefix', out, 'mapsfile', mapsfile);
info.parameters = params;         % (assigned apart: struct() would spread a struct array)
info.sidecar = sidecar;
end

% Every parameter as the fit used it, from the maps' sidecar ("Parameters", in the model's order,
% "ParameterNames"), as a struct array; `given` from "Fit", what the call set
function p = parameterlist(sc)
names = sc.ParameterNames;
if ischar(names), names = {names}; end                     % (one name decodes as text)
given = {};
if isfield(sc, 'Fit') && isstruct(sc.Fit), given = fieldnames(sc.Fit); end
p = struct('name', {}, 'role', {}, 'value', {}, 'start', {}, 'lower', {}, 'upper', {}, 'given', {});
for k = 1:numel(names)
    r = sc.Parameters.(names{k});
    e = struct('name', names{k}, 'role', r.Role, 'value', [], 'start', [], 'lower', [], ...
               'upper', [], 'given', any(strcmp(names{k}, given)));
    switch r.Role
        case 'free'
            e.start = r.Start;
            e.lower = bound(r.Lower, -Inf);
            e.upper = bound(r.Upper, Inf);
        case 'fixed'
            e.value = r.Value;
        case 'tied'
            e.value = r.To;
    end
    p(end + 1) = e; %#ok<AGROW>
end
end

function b = bound(x, none)
b = x;
if isempty(x), b = none; end                               % null in the JSON: no bound
end

% The parameters as a short table, the ones at EQUIP's defaults marked as such
function printparameters(p, model)
w = max(cellfun(@numel, {p.name}));
fprintf('parameters as fitted (* given in this call; the others %s''s defaults):\n', model);
for k = 1:numel(p)
    mark = ' ';
    if p(k).given, mark = '*'; end
    switch p(k).role
        case 'free'
            what = sprintf('free    start %-8s bounds %s to %s', show(p(k).start), ...
                           show(p(k).lower), show(p(k).upper));
        case 'fixed'
            what = sprintf('fixed   %s', show(p(k).value));
        otherwise
            what = sprintf('tied    to %s', show(p(k).value));
    end
    fprintf('  %s %-*s  %s\n', mark, w, p(k).name, what);
end
fprintf('\n');
end

function s = show(x)
if ischar(x), s = x; else, s = num2str(x); end
end

% -------------------------------------------------------------------------------------------------

% Text from char or a string scalar
function s = textarg(x, what)
if ischar(x)
    s = x;
elseif isstring(x) && isscalar(x)
    s = char(x);
else
    error('equip:fit', '%s: give it as text (''...''), not a %s', what, class(x));
end
end

% Numbers as equip's comma list, with every digit that matters (0.01, not 0.0100000000000002)
function s = numlist(v)
s = strjoin(arrayfun(@(x) sprintf('%.15g', x), double(v(:)'), 'UniformOutput', false), ',');
end

% Times are in seconds in EQUIP: values over a minute in a time are milliseconds by mistake
function checkseconds(name, v)
times = {'TI', 'TD', 'TE', 'TR', 'TM', 'TS', 'delta', 'Delta'};
if any(strcmp(name, times)) && max(abs(double(v(:)))) > 60
    error('equip:fit', ['''%s'': EQUIP takes times in seconds; these look like milliseconds ' ...
          '(divide by 1000: [%s] / 1000)'], name, num2str(v(:)'));
end
end

% The file's name without .nii, .nii.gz or .mat, with its folder: what the maps are named after
function s = stem(f)
f = regexprep(f, '(\.mat):[A-Za-z]\w*$', '$1', 'ignorecase');
s = regexprep(f, '(\.nii\.gz|\.nii|\.mat)$', '', 'ignorecase');
end

% The arrays for equip in one .mat file: version 7 (which EQUIP reads), 7.3 above 2 GB
function saveinputs(f, vars)
bytes = 0;
names = fieldnames(vars);
for k = 1:numel(names)
    x = vars.(names{k});
    info = whos('x');
    bytes = max(bytes, info.bytes);
end
if bytes > 2^31 - 1000
    save(f, '-struct', 'vars', '-v7.3');
else
    save(f, '-struct', 'vars', '-v7');
end
end
