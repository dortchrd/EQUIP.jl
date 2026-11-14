function [status, out, cmd] = equip_run(words, echo)
%EQUIP_RUN  Run EQUIP's command line with the given words (a cell array of char), as the `equip`
%   script in EQUIP.jl/cli/bin does, but calling Julia directly: so it works where MATLAB has no
%   PATH to Julia, and on Windows (no bash). Returns the exit status (0: done), everything the
%   command printed (errors included) and the command line run. With echo = true the output is
%   shown as it comes instead (for long runs) and not returned.

if nargin < 2, echo = false; end
[~, cli] = equip_julia();
cmd = equip_cmd([{'--startup-file=no', '--threads=auto', ['--project=' cli], '-m', 'EQUIPCLI'}, words]);
[status, out] = equip_system(cmd, echo);
end
