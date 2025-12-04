function [segments, labels] = prepLabeledSegment(signal, fs, evs, win, step, thr)
[nt, nch] = size(signal);
ev_msk = false(nt, 1);
nev = size(evs, 1);
for nn = 1:nev
    tev = round(evs(nn, :) * fs) + [1, 0];
    ev_msk(tev(1):tev(2)) = true;
end

nt_win = round(fs * win);
nt_step = round(fs * step);
t_ends = nt_win:nt_step:nt;
ns = length(t_ends);
ntc_start = round((nt_win - nt_step) / 2);
ntc_end = round((nt_win + nt_step) / 2) - 1;

segments = zeros(ns, nt_win, nch, 'single');
labels = zeros(ns, 1, 'single');
for nn = 1:ns
    t_end = t_ends(nn);
    t_start = t_end - nt_win + 1;
    tc_start = t_start + ntc_start;
    tc_end = t_start + ntc_end;
    segments(nn, :, :) = signal(t_start:t_end, :);
    labels(nn) = mean(ev_msk(tc_start:tc_end)) > thr;
end
end