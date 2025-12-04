%%
raw_folder = 'datasets\Bonn\Bonn Univeristy Dataset';
out_folder = 'datasets\Bonn';
sfreq = 173.6;
nt0 = 4097;
dfs = 200;
nseg = 100;
set_names = {'Z', 'O', 'N', 'F', 'S'};

ts0 = (0:1:nt0-1)' / sfreq;
ts = (0:1/dfs:ts0(end))';
nt = length(ts);
nset = length(set_names);
[segments, labels] = deal(cell(nset, 1));

for kk = 1:nset
    segments{kk} = zeros(nseg, 1, 1, nt, 'single');
    labels{kk} = ones(nseg, 1, 'single') * kk;
    for nn = 1:nseg
        sig = readmatrix(fullfile(raw_folder, set_names{kk}, ...
            sprintf('%s%.3d.txt', set_names{kk}, nn)));
        segments{kk}(nn, :, :, :) = interp1(ts0, sig, ts, 'spline');
    end
end

writeNPY(cat(1, segments{:}), fullfile(out_folder, 'all_X.npy'));
writeNPY(single(cat(1, labels{:}) == 5), fullfile(out_folder, 'all_Y.npy'));

%%
nparts = 10;
segments = readNPY(fullfile(out_folder, 'all_X.npy'));
labels = readNPY(fullfile(out_folder, 'all_Y.npy'));

for nn = 1:nparts
    mid = sprintf('P%d', nn-1);
    writeNPY(segments(nn:nparts:end, :, :, :), fullfile(out_folder, [mid, '_X.npy']));
    writeNPY(labels(nn:nparts:end), fullfile(out_folder, [mid, '_Y.npy']));
end




