function cmd = equip_cmd(args)
%EQUIP_CMD  The shell command that runs Julia with these arguments (a cell array of char): Julia
%   by its full path (equip_julia), each word quoted for the shell, single quotes on macOS and
%   Linux (nothing inside is interpreted), double quotes on Windows.

julia = equip_julia();
cmd = strjoin(cellfun(@quoted, [{julia}, args], 'UniformOutput', false), ' ');
if isunix && ~ismac
    % MATLAB on Linux puts its own libraries first on LD_LIBRARY_PATH, which can make other
    % programs load the wrong ones; Julia brings its own
    cmd = ['env -u LD_LIBRARY_PATH ' cmd];
end
end

function q = quoted(w)
if ispc
    q = ['"' w '"'];
else
    q = ['''' strrep(w, '''', '''\''''') ''''];
end
end
