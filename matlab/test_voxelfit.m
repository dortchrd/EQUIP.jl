% Tests for the MATLAB front end (runs in MATLAB or GNU Octave):  test_voxelfit
addpath(fileparts(mfilename('fullpath')));
rand('seed', 1); randn('seed', 1);

% --- MonoExp: vector protocol, mask, standard errors ---------------------------------------
te = (10:10:320)';
nx = 40; ny = 30; nz = 4;
T2 = 40 + 80*rand(nx, ny, nz); S0 = 800 + 400*rand(nx, ny, nz);
data = zeros(nx, ny, nz, numel(te));
for k = 1:numel(te)
  data(:,:,:,k) = S0 .* exp(-te(k) ./ T2) + 5*randn(nx, ny, nz);
end
mask = rand(nx, ny, nz) > 0.3;
tic; res = voxelfit('MonoExp', te, single(data), 'Mask', mask, 'StdErr', true, 'Verbose', false); t = toc;
printf('MonoExp: %.2f s including Julia start-up\n', t);
assert(isequal(res.names, {'S0', 'T2'}));
assert(isequal(size(res.T2), [nx ny nz]) && isa(res.T2, 'single'));
rel = abs(res.T2(mask) - T2(mask)) ./ T2(mask);
assert(median(rel) < 0.01);
assert(all(isnan(res.T2(~mask))) && all(res.status(~mask) == 0));
assert(all(res.status(mask) >= 1 & res.status(mask) <= 3));
assert(isfield(res, 'stderr') && isequal(size(res.stderr.T2), [nx ny nz]));

% --- SPGR with a B1 map (struct protocol, per-voxel fixed map), 2-d data -------------------
acq = struct('FA', [2 5 10 15 20 30], 'TR', 15);
n = 500;
T1 = 600 + 1000*rand(n, 1); M0 = 4000 + 2000*rand(n, 1); B1 = 0.8 + 0.4*rand(n, 1);
sig = zeros(n, numel(acq.FA));
for k = 1:numel(acq.FA)
  a = deg2rad(acq.FA(k)) * B1; E1 = exp(-acq.TR ./ T1);
  sig(:, k) = M0 .* sin(a) .* (1 - E1) ./ (1 - cos(a) .* E1) + 2*randn(n, 1);
end
r = voxelfit('SPGR', acq, sig, 'Fixed', struct('B1', B1), 'Precision', 'double', 'Verbose', false);
assert(isequal(size(r.T1), [n 1]) && isa(r.T1, 'double'));
assert(isequal(r.names, {'M0', 'T1'}));
printf('SPGR+B1 median rel T1 err %.4f\n', median(abs(r.T1 - T1) ./ T1));
assert(median(abs(r.T1 - T1) ./ T1) < 0.03);

% --- SIRFSE: kmf fixed vs estimated, Sm map, bounds ----------------------------------------
acq = struct('TI', [0.010 0.050 0.056 0.277 0.843], 'TD', [3.270 4.489 1.652 2.922 0.010]);
% noise-free SIR-FSE data from Eq. 1 of Dortch et al., MRM 2018 (independent of VoxelFit)
y = zeros(1, 5);
pmf = 0.12; R1f = 1.0; Sf = -0.95; M0f = 1000; kmf = 15; Sm = 0.83; R1m = R1f;
kfm = kmf*pmf; A = [-(R1f+kfm) kmf; kfm -(R1m+kmf)]; M0v = [M0f; pmf*M0f]; S = diag([Sf Sm]);
for k = 1:5
  Mz = (expm(A*acq.TI(k)) * S * (eye(2) - expm(A*acq.TD(k))) + (eye(2) - expm(A*acq.TI(k)))) * M0v;
  y(k) = abs(Mz(1));
end
D = repmat(reshape(y, 1, 1, 5), [4 3 1]);
r = voxelfit('SIRFSE', acq, D, 'Free', {'kmf'}, 'Bounds', struct('kmf', [1 50]), ...
             'Precision', 'double', 'Verbose', false);
assert(isequal(r.names, {'pmf', 'R1f', 'Sf', 'M0f', 'kmf'}));
assert(max(abs(r.kmf(:) - 15)) < 1e-3 && max(abs(r.pmf(:) - 0.12)) < 1e-5);
r = voxelfit('SIRFSE', acq, D, 'Fixed', struct('kmf', 15, 'Sm', 0.83*ones(4, 3)), ...
             'Precision', 'double', 'Verbose', false);
assert(isequal(r.names, {'pmf', 'R1f', 'Sf', 'M0f'}) && max(abs(r.pmf(:) - 0.12)) < 1e-5);
printf('SIRFSE: kmf free and fixed OK\n');

% --- custom expression model ----------------------------------------------------------------
m = struct('expr', 'S0*exp(-TE/T2) + C', 'params', {{'S0', 'T2', 'C'}}, 'protocol', {{'TE'}});
r2 = voxelfit(m, te, data(:,:,1,:), 'P0', [1000 60 0], 'Lower', [0 1e-3 0], 'Verbose', false);
assert(isequal(size(r2.T2), [nx ny]) && isequal(r2.names, {'S0', 'T2', 'C'}));

% --- errors propagate ------------------------------------------------------------------------
try
  voxelfit('NoSuchModel', te, data, 'Verbose', false); error('should have failed');
catch err
  assert(~isempty(strfind(err.message, 'unknown model')));
end
try
  voxelfit('SIRFSE', struct('TI', acq.TI), D, 'Verbose', false); error('should have failed');
catch err
  assert(~isempty(strfind(err.message, 'TD')));
end
disp('ALL MATLAB/OCTAVE TESTS PASSED');
