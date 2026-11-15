function cmd = equip_cmd(args)
%EQUIP_CMD  The shell command that runs Julia with these arguments (a cell array of char): Julia
%   by its full path (equip_julia), each word quoted for the shell (equip_quote).

julia = equip_julia();
cmd = strjoin(cellfun(@equip_quote, [{julia}, args], 'UniformOutput', false), ' ');
if isunix && ~ismac
    % MATLAB on Linux puts its own libraries first on LD_LIBRARY_PATH, which can make other
    % programs load the wrong ones; Julia brings its own
    cmd = ['env -u LD_LIBRARY_PATH ' cmd];
end
end
