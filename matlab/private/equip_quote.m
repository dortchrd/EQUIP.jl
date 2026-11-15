function q = equip_quote(w)
%EQUIP_QUOTE  One word of a shell command, quoted: single quotes on macOS and Linux (nothing
%   inside is interpreted), double quotes on Windows.

if ispc
    q = ['"' w '"'];
else
    q = ['''' strrep(w, '''', '''\''''') ''''];
end
end
