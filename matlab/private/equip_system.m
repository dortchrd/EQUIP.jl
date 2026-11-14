function [status, out] = equip_system(cmd, echo)
%EQUIP_SYSTEM  Run a shell command, its errors with its output; shown as it comes (echo = true,
%   then nothing returned) or returned.
%
%   On Windows the whole command goes in one more pair of double quotes: cmd.exe, which MATLAB's
%   system uses there, strips the first and the last quote of a command line that starts with
%   one, and the first word here is Julia's quoted path.

if ispc
    cmd = ['"' cmd '"'];
end
cmd = [cmd ' 2>&1'];
if echo
    if exist('OCTAVE_VERSION', 'builtin')
        status = system(cmd);
    else
        status = system(cmd, '-echo');
    end
    out = '';
else
    [status, out] = system(cmd);
end
end
