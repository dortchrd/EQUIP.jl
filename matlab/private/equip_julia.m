function [julia, cli] = equip_julia()
%EQUIP_JULIA  Where Julia is, and EQUIP's command-line package (the cli folder of EQUIP.jl).
%
%   Julia is looked for in this order, so that MATLAB finds it without the PATH of your
%   terminal (MATLAB started from the Dock or the Start menu does not have it):
%     1. the environment variable JULIA, if set: setenv('JULIA', '/path/to/julia') in startup.m
%     2. `julia` on MATLAB's own PATH
%     3. juliaup's folder (~/.juliaup/bin), where Julia's installer puts it
%     4. the usual install folders: /Applications/Julia-*.app (macOS), Homebrew, /usr/local/bin;
%        on Windows the Microsoft Store's alias and %LOCALAPPDATA%\Programs\Julia-*
%   The newest version is taken where several are installed side by side.

here = fileparts(mfilename('fullpath'));            % EQUIP.jl/matlab/private
cli = fullfile(fileparts(fileparts(here)), 'cli');  % EQUIP.jl/cli
if ~exist(fullfile(cli, 'Project.toml'), 'file')
    error('equip:nocli', ['EQUIP''s command line was not found at %s: keep the matlab ' ...
          'folder inside the EQUIP.jl folder you downloaded'], cli);
end

julia = getenv('JULIA');
if ~isempty(julia)
    if exist(julia, 'file'), return; end
    error('equip:nojulia', 'JULIA is set to %s, which does not exist', julia);
end

if ispc
    [s, out] = system('where julia');
    candidates = [firstline(s, out), ...
        {fullfile(getenv('LOCALAPPDATA'), 'Microsoft', 'WindowsApps', 'julia.exe'), ...
         fullfile(getenv('USERPROFILE'), '.juliaup', 'bin', 'julia.exe')}, ...
        newest(fullfile(getenv('LOCALAPPDATA'), 'Programs'), 'Julia-*', fullfile('bin', 'julia.exe'))];
else
    [s, out] = system('command -v julia');
    candidates = [firstline(s, out), ...
        {fullfile(getenv('HOME'), '.juliaup', 'bin', 'julia'), '/opt/homebrew/bin/julia', ...
         '/usr/local/bin/julia'}, ...
        newest('/Applications', 'Julia-*.app', fullfile('Contents', 'Resources', 'julia', 'bin', 'julia'))];
end
for k = 1:numel(candidates)
    if ~isempty(candidates{k}) && exist(candidates{k}, 'file')
        julia = candidates{k};
        return
    end
end
error('equip:nojulia', ['Julia was not found. Install it (https://julialang.org/install), ' ...
      'or, if it is installed, tell MATLAB where: setenv(''JULIA'', ''/path/to/julia'') ' ...
      '(in a terminal, `which julia` on macOS and Linux, `where julia` on Windows, shows ' ...
      'the path); put that line in startup.m to keep it']);
end

% The first line a successful `which`/`where` printed, as a one-element cell (or none)
function c = firstline(status, out)
c = {};
if status == 0
    lines = regexp(strtrim(out), '[\r\n]+', 'split');
    if ~isempty(lines) && ~isempty(lines{1}), c = {strtrim(lines{1})}; end
end
end

% The newest of the folders matching a pattern in a parent folder, with a path inside it
function c = newest(parent, pattern, inside)
c = {};
d = dir(fullfile(parent, pattern));
if isempty(d), return; end
names = sort({d.name});                 % Julia-1.10.5 < Julia-1.11.6 by version for 1.x
v = cellfun(@versionkey, names);
[~, k] = max(v);
c = {fullfile(parent, names{k}, inside)};
end

% A sortable number from the version in a folder name (Julia-1.11.6.app: 1011006)
function v = versionkey(name)
t = regexp(name, '(\d+)\.(\d+)(?:\.(\d+))?', 'tokens', 'once');
v = 0;
if isempty(t), return; end
t(cellfun(@isempty, t)) = {'0'};
n = str2double(t);
v = n(1) * 1e6 + n(2) * 1e3 + n(3);
end
