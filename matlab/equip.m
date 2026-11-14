function [status, out] = equip(cmdline)
%EQUIP  Run an equip command from MATLAB, as it would be typed in a terminal (without `equip`).
%
%   equip('fit --help')                        the models, and fit's options
%   equip('fit QMTSIR --help')                 one model's options and parameters
%   equip('info SIR.nii')                      a file's header and what EQUIP reads from it
%   equip('crlb QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I --snr 200 --kmf 12.5')
%   equip('fit QMTSIR SIR.nii --mask MASK.nii --TI 1.007,0.010,0.278,0.010 --TD 0.010,4.174,2.730,0.684')
%
%   Prints what the command prints, and stops with an error if it fails. With outputs,
%   [status, out] = equip(...) returns the exit status (0: done) and the output instead.
%   A word with spaces (a folder's name) goes in double quotes: equip('info "My data/SIR.nii"').
%   For fits into MATLAB variables, equip_fit. Times in seconds.
%
%   See also EQUIP_FIT, EQUIP_SETUP.

if ~(ischar(cmdline) || (isstring(cmdline) && isscalar(cmdline)))
    error('equip:args', 'give the command as text: equip(''fit QMTSIR --help'')');
end
words = regexp(char(cmdline), '"[^"]*"|\S+', 'match');
words = regexprep(words, '^"(.*)"$', '$1');
if ~isempty(words) && strcmp(words{1}, 'equip'), words(1) = []; end
[s, txt] = equip_run(words);
if nargout > 0
    status = s;
    out = txt;
    return
end
fprintf('%s\n', strtrim(txt));
if s ~= 0
    error('equip:failed', 'equip stopped (exit status %d): see its message above', s);
end
end
