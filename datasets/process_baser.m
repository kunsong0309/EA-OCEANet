%%
raw_folder = 'datasets\Baser\Baser2022\ALL_RATS_RAW_DATA\MAT';
out_folder = 'datasets\Baser';

fls = dir(fullfile(raw_folder, 'labels_*.mat'));
dfs = 200;
for nn = 1:length(fls)
    mid = regexp(fls(nn).name, '\d+(?=.mat)', 'match', 'once');
    load(fullfile(fls(nn).folder, fls(nn).name), 'Final_labels');
    load(fullfile(fls(nn).folder, ['Voltage_Animal', mid, 'CH1.mat']), 'Voltage_CH1');
    load(fullfile(fls(nn).folder, ['Voltage_Animal', mid, 'CH2.mat']), 'Voltage_CH2');

    segments = permute(cat(3, Voltage_CH1, Voltage_CH2), [1, 4, 3, 2]);
    writeNPY(single(segments(:, :, :, 3:(1000/dfs):end)), ...
        fullfile(out_folder, [mid, '_X.npy']));
    writeNPY(single(Final_labels(1, :)'), fullfile(out_folder, [mid, '_Y.npy'])); 
end

