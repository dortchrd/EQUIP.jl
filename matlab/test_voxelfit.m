addpath(fileparts(mfilename('fullpath')));
rand('seed', 1); randn('seed', 1);
te = (10:10:320)';
nx = 40; ny = 30; nz = 4;
T2 = 40 + 80*rand(nx, ny, nz); S0 = 800 + 400*rand(nx, ny, nz);
data = zeros(nx, ny, nz, numel(te));
for k = 1:numel(te)
  data(:,:,:,k) = S0 .* exp(-te(k) ./ T2) + 5*randn(nx, ny, nz);
end
mask = rand(nx, ny, nz) > 0.3;
tic; res = voxelfit('MonoExp', te, single(data), 'Mask', mask, 'StdErr', true); t = toc;
printf('wall time incl. Julia start-up: %.2f s\n', t);
disp(fieldnames(res)')
assert(isequal(size(res.T2), [nx ny nz]));
assert(isa(res.T2, 'single'));
rel = abs(res.T2(mask) - T2(mask)) ./ T2(mask);
printf('median rel T2 err %.4f\n', median(rel));
assert(median(rel) < 0.01);
assert(all(isnan(res.T2(~mask))) && all(res.status(~mask) == 0));
assert(all(res.status(mask) >= 1 & res.status(mask) <= 3));
% SPGR with B1 covariate, double precision, 2-d data (voxels x observations)
fa = [2 5 10 15 20 30]'; TR = 15; n = 500;
T1 = 600 + 1000*rand(n, 1); M0 = 4000 + 2000*rand(n, 1); B1 = 0.8 + 0.4*rand(n, 1);
sig = zeros(n, numel(fa));
for k = 1:numel(fa)
  a = deg2rad(fa(k)) * B1; E1 = exp(-TR ./ T1);
  sig(:, k) = M0 .* sin(a) .* (1 - E1) ./ (1 - cos(a) .* E1) + 2*randn(n, 1);
end
r = voxelfit('SPGR(TR=15)', fa, sig, 'Covariates', B1, 'Precision', 'double', 'Verbose', false);
assert(isequal(size(r.T1), [n 1]) && isa(r.T1, 'double'));
printf('SPGR+B1 median rel T1 err %.4f\n', median(abs(r.T1 - T1) ./ T1));
% custom expression model
m.expr = 'S0*exp(-TE/T2) + C'; m.params = {'S0', 'T2', 'C'}; m.xname = 'TE';
r2 = voxelfit(m, te, data(:,:,1,:), 'P0', [1000 60 0], 'Lower', [0 1e-3 0], 'Verbose', false);
assert(isequal(size(r2.T2), [nx ny]));
printf('expr model fields: %s\n', strjoin(r2.names, ', '));
% error path
try
  voxelfit('NoSuchModel', te, data, 'Verbose', false); error('should have failed');
catch err
  assert(~isempty(strfind(err.message, 'unknown model')));
  disp('error propagated OK');
end
disp('ALL MATLAB/OCTAVE TESTS PASSED');
