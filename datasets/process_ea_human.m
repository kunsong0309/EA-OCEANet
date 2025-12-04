%%
raw_folder = 'dataset\EA-Human\sEEG';
out_folder = 'datasets\EA-Human';
fls = dir(fullfile(raw_folder, '*.mat'));
dfs = 200;
freqr = [0.1, 100];
win = 1;
step = 0.5;
thr = 0.1;
for nn = 1:length(fls)
    mid = num2str(nn - 1, '1%d');
    load(fullfile(fls(nn).folder, fls(nn).name), 'eeg', 'fs', 'evs');

    if mad(eeg) < 1
        scal = 1e3;
    else
        scal = 1;
    end
    signal = prepSignal(eeg * scal, fs, dfs, freqr);

    [segments, labels] = prepLabeledSegment(signal, dfs, evs, win, step, thr);
    t1 = find(labels == 1, 1, "first");
    t2 = find(labels == 1, 1, "last");
    segments = segments(t1:t2, :);
    labels = labels(t1:t2);
    writeNPY(permute(segments, [1, 3, 4, 2]), fullfile(out_folder, [mid, '_X.npy']));
    writeNPY(labels, fullfile(out_folder, [mid, '_Y.npy'])); 
end