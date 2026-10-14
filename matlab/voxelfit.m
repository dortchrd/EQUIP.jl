function res = voxelfit(model, x, data, varargin)
%VOXELFIT  Voxel-wise nonlinear least-squares fitting of MRI data with VoxelFit.jl.
%
%   res = voxelfit(model, x, data)
%   res = voxelfit(model, x, data, 'Name', Value, ...)
%
%   model  Built-in model name, e.g. 'MonoExp', 'SPGR(TR=15)', 'IVIM', 'InversionRecovery'
%          (run "voxelfit('models')" to list them), or a struct describing a custom model
%          as a Julia expression:
%             m.expr   = 'S0*exp(-TE/T2) + C';
%             m.params = {'S0','T2','C'};
%             m.xname  = 'TE';              % optional, default 'x' (cellstr for several)
%             m.covariates = {'B1'};        % optional names for 'Covariates'
%   x      Acquisition values, one row per observation (nobs x 1, or nobs x k).
%   data   Array with the observations along the LAST dimension, e.g. X x Y x Z x nobs.
%
%   Options (name/value):
%     'Mask'        array of size(data) without the last dimension; zeros are skipped
%     'Covariates'  per-voxel maps passed to the model: spatial size, or [spatial K]
%                   (e.g. a relative B1 map for SPGR)
%     'P0'          initial guess for every voxel (1 x P); default: data-driven per voxel
%     'Lower', 'Upper'  bounds (1 x P, use -Inf/Inf for none)
%     'Weights'     per-observation weights (nobs x 1)
%     'Precision'   'single' (default) or 'double'
%     'StdErr'      true to also return standard-error maps (default false)
%     'MaxIter', 'XTol', 'FTol', 'GTol'   solver settings
%     'Backend'     'cpu' (default), 'cuda', 'amdgpu', 'metal'
%     'Threads'     number of Julia threads or 'auto' (default)
%     'Julia'       julia executable (default: $VOXELFIT_JULIA or 'julia')
%     'Project'     path to VoxelFit.jl (default: the folder above this file)
%     'Sysimage'    optional system image for fast start-up
%     'Verbose'     print the Julia summary (default true)
%
%   res    struct with one field per parameter (maps of size(data) without the last
%          dimension), plus sse, iterations, status, names and, with 'StdErr', stderr.
%          status: 0 not fitted, 1-3 converged, 4 max iterations, 5 failed, 6 invalid data.
%
%   Example:
%     te  = (10:10:320)';
%     res = voxelfit('MonoExp', te, img, 'Mask', brain);
%     imagesc(res.T2(:,:,10), [0 150]); axis image; colorbar
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
if size(x, 1) ~= nobs
    if isvector(x) && numel(x) == nobs
        x = x(:);
    else
        error('voxelfit:x', 'x has %d rows but data have %d observations', size(x, 1), nobs);
    end
end
if ~isa(data, 'single') && ~isa(data, 'double') && ~isinteger(data)
    data = double(data);
end
args = {'fit', '--data', write_npy(fullfile(tmp, 'data.npy'), data), ...
        '--x-file', write_txt(fullfile(tmp, 'x.txt'), x), ...
        '--out', fullfile(tmp, 'out'), '--format', 'npy', '--precision', opts.precision};

if isstruct(model)
    args = [args, {'--expr', model.expr, '--params', join_list(model.params)}];
    if isfield(model, 'xname'), args = [args, {'--xname', join_list(model.xname)}]; end
    if isfield(model, 'covariates')
        args = [args, {'--covariate-names', join_list(model.covariates)}];
    end
else
    args = [args, {'--model', model}];
end
if ~isempty(opts.mask)
    check_spatial(opts.mask, spatial, 'Mask');
    args = [args, {'--mask', write_npy(fullfile(tmp, 'mask.npy'), uint8(opts.mask ~= 0))}];
end
if ~isempty(opts.covariates)
    C = opts.covariates;
    csz = size(C);
    if isequal(csz(1:min(end, numel(spatial))), spatial) && numel(C) > prod(spatial)
        K = csz(end);
        C = reshape(C, [prod(spatial), K]);
    else
        check_spatial(C, spatial, 'Covariates');
        K = 1;
        C = C(:);
    end
    for k = 1:K
        f = write_npy(fullfile(tmp, sprintf('cov%d.npy', k)), reshape(double(C(:, k)), [spatial 1]));
        args = [args, {'--covariate', f}]; %#ok<AGROW>
    end
end
if ~isempty(opts.p0),      args = [args, {'--p0', num_list(opts.p0)}]; end
if ~isempty(opts.lower),   args = [args, {'--lower', num_list(opts.lower)}]; end
if ~isempty(opts.upper),   args = [args, {'--upper', num_list(opts.upper)}]; end
if ~isempty(opts.weights), args = [args, {'--weights', write_txt(fullfile(tmp, 'w.txt'), opts.weights(:))}]; end
if opts.stderr,            args = [args, {'--stderr'}]; end
if ~isempty(opts.maxiter), args = [args, {'--maxiter', sprintf('%d', opts.maxiter)}]; end
if ~isempty(opts.xtol),    args = [args, {'--xtol', num_list(opts.xtol)}]; end
if ~isempty(opts.ftol),    args = [args, {'--ftol', num_list(opts.ftol)}]; end
if ~isempty(opts.gtol),    args = [args, {'--gtol', num_list(opts.gtol)}]; end
if ~strcmpi(opts.backend, 'cpu'), args = [args, {'--backend', opts.backend}]; end
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
opts = struct('mask', [], 'covariates', [], 'p0', [], 'lower', [], 'upper', [], ...
    'weights', [], 'precision', 'single', 'stderr', false, 'maxiter', [], 'xtol', [], ...
    'ftol', [], 'gtol', [], 'backend', 'cpu', 'threads', 'auto', 'julia', '', ...
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
