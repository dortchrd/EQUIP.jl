function res = voxelfit(model, acq, data, varargin)
%VOXELFIT  Voxel-wise nonlinear least-squares fitting of MRI data with VoxelFit.jl.
%
%   res = voxelfit(model, acq, data)
%   res = voxelfit(model, acq, data, 'Name', Value, ...)
%
%   model  Built-in model name, e.g. 'MonoExp', 'SPGR', 'SIRFSE', 'SIRFSE(magnitude=false)'
%          (run "voxelfit('models')" to list them with their protocol and parameters), or a
%          struct describing a custom model as a Julia expression for one observation:
%             m.expr     = 'S0*exp(-TE/T2) + C';
%             m.params   = {'S0','T2','C'};
%             m.protocol = {'TE'};           % optional, default {'x'}
%   acq    Protocol (acquisition settings). A vector for one-setting models (e.g. echo
%          times for MonoExp), or a struct whose fields are vectors (one value per
%          observation) or scalars (constant), e.g. acq.TI = [...]; acq.TD = 2.5;
%   data   Array with the observations along the LAST dimension, e.g. X x Y x Z x nobs.
%
%   Options (name/value):
%     'Mask'      array of size(data) without the last dimension; zeros are skipped
%     'Fixed'     struct of parameters to hold fixed: scalars, or per-voxel maps with the
%                 spatial size, e.g. struct('kmf', 12.5, 'Sm', smMap) or struct('B1', b1)
%     'Free'      parameters the model fixes by default that should be estimated,
%                 e.g. {'kmf'}
%     'Start'     struct of starting values (scalars or maps) for estimated parameters
%     'Bounds'    struct of [lower upper] per estimated parameter, e.g. struct('kmf', [1 50])
%     'P0', 'Lower', 'Upper'   starting values and bounds for an expression model (1 x P)
%     'Weights'   per-observation weights (nobs x 1)
%     'Precision' 'single' (default) or 'double'
%     'StdErr'    true to also return standard-error maps (default false)
%     'MaxIter', 'XTol', 'FTol', 'GTol'   solver settings
%     'Threads'   number of Julia threads or 'auto' (default)
%     'Julia'     julia executable (default: $VOXELFIT_JULIA or 'julia')
%     'Project'   path to VoxelFit.jl (default: the folder above this file)
%     'Sysimage'  optional system image for fast start-up
%     'Verbose'   print the Julia summary (default true)
%
%   res    struct with one field per estimated parameter (maps of size(data) without the
%          last dimension), plus sse, iterations, status, names and, with 'StdErr', stderr.
%          status: 0 not fitted, 1-3 converged, 4 max iterations, 5 failed,
%          6 invalid data or fixed-map value.
%
%   Examples:
%     res = voxelfit('MonoExp', (10:10:320)', img, 'Mask', brain);
%     acq = struct('TI', ti, 'TD', td);
%     res = voxelfit('SIRFSE', acq, img, 'Fixed', struct('kmf', 12.5), 'Mask', brain);
%     res = voxelfit('SPGR', struct('FA', fa, 'TR', 15), vfa, 'Fixed', struct('B1', b1));
%
%   Data are exchanged through temporary .npy files and Julia is started for each call
%   (about 1-3 s of overhead), so fit whole volumes per call rather than single voxels.

if ischar(model) && strcmpi(model, 'models') && nargin == 1
    run_cli(default_opts(), {'models'}, true);
    return
end

opts = parse_options(default_opts(), varargin);
tmp = tempname();
mkdir(tmp);
cleaner = onCleanup(@() rmdir_quiet(tmp));

% ---- inputs ------------------------------------------------------------------------
sz = size(data);
nobs = sz(end);
spatial = sz(1:end-1);
if ~isa(data, 'single') && ~isa(data, 'double') && ~isinteger(data)
    data = double(data);
end
args = {'fit', '--data', write_npy(fullfile(tmp, 'data.npy'), data), ...
        '--out', fullfile(tmp, 'out'), '--format', 'npy', '--precision', opts.precision};

% protocol
if isstruct(acq)
    names = fieldnames(acq);
    for i = 1:numel(names)
        v = acq.(names{i});
        if ~isscalar(v) && numel(v) ~= nobs
            error('voxelfit:acq', 'acq.%s has %d values but data have %d observations', ...
                  names{i}, numel(v), nobs);
        end
        args = [args, {'--acq', [names{i} '=' num_list(v)]}]; %#ok<AGROW>
    end
else
    if numel(acq) ~= nobs
        error('voxelfit:acq', 'acq has %d values but data have %d observations', numel(acq), nobs);
    end
    args = [args, {'--x-file', write_txt(fullfile(tmp, 'x.txt'), acq(:))}];
end

% model
if isstruct(model)
    args = [args, {'--expr', model.expr, '--params', join_list(model.params)}];
    if isfield(model, 'protocol'), args = [args, {'--protocol', join_list(model.protocol)}]; end
else
    args = [args, {'--model', model}];
end

% parameters
args = [args, value_args('--fixed', opts.fixed, spatial, tmp, 'fixed')];
args = [args, value_args('--start', opts.start, spatial, tmp, 'start')];
if ~isempty(opts.free), args = [args, {'--free', join_list(opts.free)}]; end
if ~isempty(opts.bounds)
    names = fieldnames(opts.bounds);
    for i = 1:numel(names)
        args = [args, {'--bounds', [names{i} '=' num_list(opts.bounds.(names{i}))]}]; %#ok<AGROW>
    end
end
if ~isempty(opts.p0),      args = [args, {'--p0', num_list(opts.p0)}]; end
if ~isempty(opts.lower),   args = [args, {'--lower', num_list(opts.lower)}]; end
if ~isempty(opts.upper),   args = [args, {'--upper', num_list(opts.upper)}]; end

% other options
if ~isempty(opts.mask)
    check_spatial(opts.mask, spatial, 'Mask');
    args = [args, {'--mask', write_npy(fullfile(tmp, 'mask.npy'), uint8(opts.mask ~= 0))}];
end
if ~isempty(opts.weights), args = [args, {'--weights', write_txt(fullfile(tmp, 'w.txt'), opts.weights(:))}]; end
if opts.stderr,            args = [args, {'--stderr'}]; end
if ~isempty(opts.maxiter), args = [args, {'--maxiter', sprintf('%d', opts.maxiter)}]; end
if ~isempty(opts.xtol),    args = [args, {'--xtol', num_list(opts.xtol)}]; end
if ~isempty(opts.ftol),    args = [args, {'--ftol', num_list(opts.ftol)}]; end
if ~isempty(opts.gtol),    args = [args, {'--gtol', num_list(opts.gtol)}]; end
if ~opts.verbose,          args = [args, {'--quiet'}]; end

run_cli(opts, args, opts.verbose);

% ---- outputs -----------------------------------------------------------------------
out = fullfile(tmp, 'out');
names = strsplit(strtrim(fileread([out '_params.txt'])));
res = struct();
for i = 1:numel(names)
    res.(names{i}) = fit_shape(read_npy([out '_' names{i} '.npy']), spatial);
end
res.sse = fit_shape(read_npy([out '_sse.npy']), spatial);
res.iterations = fit_shape(read_npy([out '_iterations.npy']), spatial);
res.status = fit_shape(read_npy([out '_status.npy']), spatial);
res.names = names;
if opts.stderr
    res.stderr = struct();
    for i = 1:numel(names)
        res.stderr.(names{i}) = fit_shape(read_npy([out '_stderr_' names{i} '.npy']), spatial);
    end
end
end

% =====================================================================================

function opts = default_opts()
here = fileparts(mfilename('fullpath'));
opts = struct('mask', [], 'fixed', [], 'free', [], 'start', [], 'bounds', [], 'p0', [], ...
    'lower', [], 'upper', [], 'weights', [], 'precision', 'single', 'stderr', false, ...
    'maxiter', [], 'xtol', [], 'ftol', [], 'gtol', [], 'threads', 'auto', 'julia', '', ...
    'project', fileparts(here), 'sysimage', '', 'verbose', true);
j = getenv('VOXELFIT_JULIA');
if isempty(j), j = 'julia'; end
opts.julia = j;
end

function opts = parse_options(opts, args)
if mod(numel(args), 2) ~= 0
    error('voxelfit:options', 'options must be name/value pairs');
end
for i = 1:2:numel(args)
    name = lower(args{i});
    if ~isfield(opts, name)
        error('voxelfit:options', 'unknown option ''%s''', args{i});
    end
    opts.(name) = args{i+1};
end
if isnumeric(opts.threads), opts.threads = sprintf('%d', opts.threads); end
end

% Turn a struct of scalars / per-voxel maps into repeated NAME=VALUE (or NAME=file.npy) options.
function args = value_args(flag, s, spatial, tmp, tag)
args = {};
if isempty(s), return; end
names = fieldnames(s);
for i = 1:numel(names)
    v = s.(names{i});
    if isscalar(v)
        args = [args, {flag, [names{i} '=' num_list(v)]}]; %#ok<AGROW>
    else
        check_spatial(v, spatial, [tag '.' names{i}]);
        f = write_npy(fullfile(tmp, sprintf('%s_%s.npy', tag, names{i})), double(v));
        args = [args, {flag, [names{i} '=' f]}]; %#ok<AGROW>
    end
end
end

function run_cli(opts, args, verbose)
cmd = [q(opts.julia) ' --startup-file=no --threads=' opts.threads ' --project=' q(opts.project)];
if ~isempty(opts.sysimage), cmd = [cmd ' --sysimage=' q(opts.sysimage)]; end
cmd = [cmd ' -e "using VoxelFit; exit(VoxelFit.main(ARGS))" --'];
for i = 1:numel(args)
    cmd = [cmd ' ' q(args{i})]; %#ok<AGROW>
end
cmd = [cmd ' 2>&1'];
% MATLAB prepends its own libraries to the loader path; do not let Julia pick them up.
if ismac
    cmd = ['env -u DYLD_LIBRARY_PATH ' cmd];
elseif isunix
    cmd = ['env -u LD_LIBRARY_PATH ' cmd];
end
[status, output] = system(cmd);
if status ~= 0
    error('voxelfit:julia', 'VoxelFit failed (exit %d):\n%s\ncommand: %s', status, output, cmd);
end
if verbose
    fprintf('%s', output);
end
end

function s = q(s)
% Quote a command-line argument for the system shell.
if ispc
    s = ['"' strrep(s, '"', '\"') '"'];
else
    s = ['''' strrep(s, '''', '''\''''') ''''];
end
end

function s = join_list(c)
if ischar(c), s = c; return; end
s = sprintf('%s,', c{:});
s = s(1:end-1);
end

function s = num_list(v)
s = sprintf('%.17g,', v);
s = s(1:end-1);
end

function check_spatial(A, spatial, name)
sa = size(A);
if numel(A) ~= prod(spatial) || ~isequal(sa(1:min(end, numel(spatial))), spatial(1:min(end, numel(sa))))
    error('voxelfit:size', '%s must have size [%s]', name, num2str(spatial));
end
end

function A = fit_shape(A, spatial)
if numel(spatial) == 1
    A = reshape(A, [spatial 1]);
else
    A = reshape(A, spatial);
end
end

function f = write_txt(f, x)
fid = fopen(f, 'w');
fmt = [repmat('%.17g ', 1, size(x, 2)) '\n'];
fprintf(fid, fmt, double(x).');
fclose(fid);
end

function rmdir_quiet(d)
if exist(d, 'dir')
    rmdir(d, 's');
end
end

% ---- minimal .npy I/O (Fortran order, little-endian) ----------------------------------

function f = write_npy(f, A)
if islogical(A), A = uint8(A); end
switch class(A)
    case 'double', d = '<f8';
    case 'single', d = '<f4';
    case 'uint8',  d = '|u1';
    case 'int8',   d = '|i1';
    case 'int16',  d = '<i2';
    case 'uint16', d = '<u2';
    case 'int32',  d = '<i4';
    case 'uint32', d = '<u4';
    case 'int64',  d = '<i8';
    case 'uint64', d = '<u8';
    otherwise, error('voxelfit:npy', 'unsupported class %s', class(A));
end
sz = size(A);
shape = sprintf('%d, ', sz);
hdr = sprintf('{''descr'': ''%s'', ''fortran_order'': True, ''shape'': (%s), }', d, shape(1:end-2));
pad = mod(64 - mod(10 + numel(hdr) + 1, 64), 64);
hdr = [hdr repmat(' ', 1, pad) char(10)];
fid = fopen(f, 'w', 'ieee-le');
fwrite(fid, uint8([147 78 85 77 80 89 1 0]), 'uint8');
fwrite(fid, numel(hdr), 'uint16');
fwrite(fid, hdr, 'char');
fwrite(fid, A, class(A));
fclose(fid);
end

function A = read_npy(f)
fid = fopen(f, 'r', 'ieee-le');
if fid < 0, error('voxelfit:npy', 'cannot open %s', f); end
magic = fread(fid, 6, 'uint8=>uint8');
if ~isequal(magic(:)', uint8([147 78 85 77 80 89]))
    fclose(fid);
    error('voxelfit:npy', '%s is not a .npy file', f);
end
ver = fread(fid, 2, 'uint8');
if ver(1) == 1
    hlen = fread(fid, 1, 'uint16');
else
    hlen = fread(fid, 1, 'uint32');
end
hdr = fread(fid, hlen, 'char=>char')';
descr = regexp(hdr, '''descr''\s*:\s*''[<>|=]?(\w\d+)''', 'tokens', 'once');
fo = regexp(hdr, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
sh = regexp(hdr, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
dims = sscanf(strrep(sh{1}, ',', ' '), '%d')';
types = {'f8', 'double'; 'f4', 'single'; 'u1', 'uint8'; 'i1', 'int8'; 'i2', 'int16'; ...
         'u2', 'uint16'; 'i4', 'int32'; 'u4', 'uint32'; 'i8', 'int64'; 'u8', 'uint64'; 'b1', 'uint8'};
k = find(strcmp(types(:, 1), descr{1}), 1);
if isempty(k), fclose(fid); error('voxelfit:npy', 'unsupported dtype %s', descr{1}); end
A = fread(fid, prod(dims), ['*' types{k, 2}]);
fclose(fid);
if isempty(dims), return; end
if numel(dims) == 1, dims = [dims 1]; end
if strcmp(fo{1}, 'True')
    A = reshape(A, dims);
else
    A = permute(reshape(A, fliplr(dims)), numel(dims):-1:1);
end
end
