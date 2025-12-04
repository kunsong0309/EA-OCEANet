function signal = prepSignal(eeg, fs, dfs, freqr)
if nargin == 4 && ~isempty(freqr)
    h = design(fdesign.bandpass('N,F3dB1,F3dB2', ...
        4, min(freqr), max(freqr), fs), 'butter');
    signal = filtfilt(h.sosMatrix, h.ScaleValues, eeg);
else
    signal = eeg;
end

ts0 = (0:1:size(eeg, 1)-1)'/fs;
ts = (0:1/dfs:ts0(end))';
signal = interp1(ts0, signal, ts, 'linear', 'extrap');
end