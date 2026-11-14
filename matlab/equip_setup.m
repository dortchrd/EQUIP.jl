function ok = equip_setup()
%EQUIP_SETUP  Get EQUIP ready to use from MATLAB: once after downloading EQUIP, and again after
%   each update.
%
%   1. finds Julia (see README.md if it is not found);
%   2. installs EQUIP's Julia packages: the first time 5 to 15 minutes, with internet, while
%      MATLAB shows Julia's progress and is busy; later runs, seconds;
%   3. fits a small simulated T2 decay through equip_fit, to check that everything works;
%   4. says what to put in startup.m, so that MATLAB finds EQUIP's functions every time.
%
%   ok = equip_setup() returns true when every step worked.
%
%   See also EQUIP_FIT, EQUIP.

ok = false;
here = fileparts(mfilename('fullpath'));
addpath(here);                                   % this session; startup.m for the next ones
fprintf('\nEQUIP setup\n\n');

% 1. Julia
[julia, cli] = equip_julia();
[s, v] = equip_system(equip_cmd({'--version'}), false);
t = regexp(v, 'version (\d+)\.(\d+)\.(\d+)', 'tokens', 'once');
if s ~= 0 || isempty(t)
    error('equip:setup', 'Julia at %s did not start:\n%s', julia, v);
end
n = str2double(t);
fprintf('1. Julia %d.%d.%d, at %s\n', n, julia);
if n(1) < 1 || (n(1) == 1 && n(2) < 11)
    error('equip:setup', ['EQUIP needs Julia 1.11 or newer; this is %d.%d.%d. Update it ' ...
          '(juliaup: `juliaup update` in a terminal; or a new version from ' ...
          'https://julialang.org/downloads)'], n);
end

% 2. EQUIP's packages, in the command line's environment (EQUIP.jl/cli): resolved against this
%    copy of EQUIP (after an update the package list may have changed), downloaded, and compiled
fprintf(['\n2. Installing EQUIP''s Julia packages into %s\n   (the first time 5 to 15 ' ...
         'minutes; MATLAB is busy until it finishes)\n\n'], cli);
code = 'using Pkg; Pkg.resolve(); Pkg.instantiate(); using EQUIPCLI';   % (no quotes inside: Windows)
s = equip_system(equip_cmd({'--startup-file=no', ['--project=' cli], '-e', code}), true);
if s ~= 0
    error('equip:setup', ['installing EQUIP''s packages stopped (see Julia''s message above). ' ...
          'If it could not download, check the internet connection and run equip_setup again']);
end

% 3. A fit, end to end: T2 decay without noise, 2 x 2 x 1 voxels, 8 echoes; the fit must give
%    the T2s back
fprintf('\n3. A test fit (simulated T2 decay)\n');
te = (1:8) * 0.01;
T2 = reshape([0.04 0.06 0.08 0.10], 2, 2);
data = zeros(2, 2, 1, 8);
for k = 1:8
    data(:, :, 1, k) = 1000 * exp(-te(k) ./ T2);
end
maps = equip_fit('T2Decay', data, 'TE', te, 'verbose', false);
err = max(abs(double(maps.T2map(:)) - T2(:)) ./ T2(:));
if err > 1e-4
    error('equip:setup', 'the test fit gave T2 %s instead of %s', mat2str(maps.T2map(:)', 4), mat2str(T2(:)'));
end
fprintf('   T2 fitted as simulated (largest error %.1g): EQUIP works from MATLAB.\n', err);

% 4. startup.m
try
    up = regexprep(userpath, [pathsep '$'], '');   % MATLAB's own folder (not in Octave)
catch
    up = '';
end
if isempty(up), up = 'your MATLAB folder: run userpath to see it'; end
fprintf(['\n4. So that MATLAB finds EQUIP every time, put this line in startup.m (in %s;\n' ...
         '   open it with edit(fullfile(userpath, ''startup.m''))):\n\n' ...
         '       addpath(''%s'')\n\n'], up, here);
if ~isempty(getenv('JULIA'))
    fprintf(['   and, since Julia was found through JULIA, that line too:\n\n' ...
             '       setenv(''JULIA'', ''%s'')\n\n'], julia);
end
fprintf('Done. Try: help equip_fit\n\n');
ok = true;
end
