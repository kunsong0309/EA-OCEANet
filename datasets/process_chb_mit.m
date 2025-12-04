%%
raw_folder = 'datasets\CHB-MIT\CHB-MIT';
sfreq = 256;
dfs = 200;
thr = 0.4;
chs = {'FP1F7', 'F7T7', 'T7P7', 'P7O1', 'FP1F3', 'F3C3', 'C3P3', 'P3O1', ...
    'FP2F4', 'F4C4', 'C4P4', 'P4O2', 'FP2F8', 'F8T8', 'T8P8', 'P8O2', ...
    'FZCZ', 'CZPZ'};

%% for meta info
fls = dir(fullfile(raw_folder, 'chb*'));
for nn = 1:length(fls)
    subls = dir(fullfile(raw_folder, fls(nn).name, '*.edf'));
    lines = readlines(fullfile(fls(nn).folder, fls(nn).name, [fls(nn).name, '-summary.txt']));

    chs = cell(100, 1);
    nch = 0;
    freq = nan;
    sumls = struct;
    nf = 0;
    nev = 0;
    for kk = 1:length(lines)
        tmp = str2double(regexp(lines(kk), ...
            '(?<=Data Sampling Rate: )\d+(?= Hz)', 'match', 'once'));
        if ~isnan(tmp); freq = tmp; continue; end
       
        tmp = regexp(char(lines(kk)), '(?<=Channel \d+: )\S+', 'match', 'once');
        if any(tmp); nch = nch + 1; chs{nch} = tmp; continue; end

        tmp = regexp(char(lines(kk)), '(?<=File Name: )\S+', 'match', 'once');
        if any(tmp); nf = nf + 1; nev = 0; sumls(nf, 1).name = tmp; continue; end

        tmp = regexp(char(lines(kk)), '(?<=File Start Time: )\S+', 'match', 'once');
        if any(tmp); sumls(nf, 1).t1 = tmp; continue; end

        tmp = regexp(char(lines(kk)), '(?<=File End Time: )\S+', 'match', 'once');
        if any(tmp); sumls(nf, 1).t2 = tmp; continue; end

        tmp = regexp(char(lines(kk)), '(?<=Number of Seizures in File: )\d+', 'match', 'once');
        if any(tmp); sumls(nf, 1).evs = nan(str2double(tmp), 2); continue; end

        tmp = regexp(char(lines(kk)), '(?<=Seizure.*Start Time: ).+(?= seconds)', 'match', 'once');
        if any(tmp); nev = nev + 1; sumls(nf, 1).evs(nev, 1) = str2double(strip(tmp)); continue; end

        tmp = regexp(char(lines(kk)), '(?<=Seizure.*End Time: ).+(?= seconds)', 'match', 'once');
        if any(tmp); sumls(nf, 1).evs(nev, 2) = str2double(strip(tmp)); continue; end

    end
    chs = chs(1:nch);

    if length(sumls) ~= length(subls)
        fprintf('%s files list not match\n', fls(nn).name);
    end

    for ii = 1:length(subls)
        hdr = readEDF_x(fullfile(raw_folder, fls(nn).name, subls(ii).name));
        ids = arrayfun(@(x)(strcmp(x.name, subls(ii).name)), sumls);
        if any(ids)
            subls(ii).evs = sumls(ids).evs;
        else
            fprintf('%s file name not match\n', subls(ii).name);
        end  

        if any(ids) && isfield(sumls, 't1')
            subls(ii).t1 = duration(sumls(ids).t1);
            subls(ii).t2 = duration(sumls(ids).t2);
        else
            subls(ii).t1 = duration(strrep(hdr.starttime, '.', ':'));
            subls(ii).t2 = subls(ii).t1 + seconds(hdr.records);
        end
                       
        % if ~all(cellfun(@(x, y)(strcmp(x, y)), hdr.label(:), strrep(chs, '-', '')))
        %     fprintf('%s channels not aligned\n', subls(ii).name);
        % end
        
        subls(ii).chs = hdr.label(:);
        subls(ii).date = hdr.startdate;
        subls(ii).freq = hdr.frequency(1);
        if subls(ii).t1 ~= duration(strrep(hdr.starttime, '.', ':'))
            fprintf('%s start time not the same\n', subls(ii).name);
        end
        if seconds(subls(ii).t2 - subls(ii).t1) ~= hdr.records
            fprintf('%s duration not the same\n', subls(ii).name);
        end

        if hdr.frequency(1) ~= freq
            fprintf('%s sampling rate not the same\n', subls(ii).name);
        end
    end

    fls(nn).chs = chs;
    fls(nn).summary = subls;
end

%% for test
out_folder = 'datasets\CHB-MIT\test';
win = 10;
step = 10;
for nn = 1:length(fls)
    subls = fls(nn).summary;
    for ii = 1:length(subls)
        try
            [hdr, da] = readEDF_x(fullfile(subls(ii).folder, subls(ii).name));
        catch ER
            disp(ER);
            [hdr, da] = readEDF(fullfile(subls(ii).folder, subls(ii).name));
            da = single(da)';
        end
        if nn == 12 && ismember(ii, [11, 12, 13])
            ich = [5, 1; 1, 2; 2, 3; 3, 9; 5, 6; 6, 7; 7, 8; 8, 9; ...
                15, 16; 16, 17; 17, 18; 18, 19; 15, 21; 21, 22; ...
                22, 23; 23, 19; 11, 12; 12, 13];
            signal = prepSignal(da(:, ich(:, 1)) - da(:, ich(:, 2)), sfreq, dfs);
        else
            [~, ich] = ismember(chs, hdr.label(:));
            signal = prepSignal(da(:, ich), sfreq, dfs);
        end
        [segments, labels] = prepLabeledSegment(signal, dfs, subls(ii).evs, win, step, thr);
        subls(ii).X = segments;
        subls(ii).Y = labels;
    end

    segments = permute(cat(1, subls(:).X), [1, 4, 3, 2]);
    labels = cat(1, subls(:).Y);
    writeNPY(segments, fullfile(out_folder, [fls(nn).name, '_X.npy']));
    writeNPY(labels, fullfile(out_folder, [fls(nn).name, '_Y.npy']));
end

%% for training
out_folder = 'datasets\CHB-MIT\training';
win = 10;
step = 1;
for nn = 1:length(fls)
    subls = fls(nn).summary;
    for ii = 1:length(subls)
        try
            [hdr, da] = readEDF_x(fullfile(subls(ii).folder, subls(ii).name));
        catch ER
            disp(ER);
            [hdr, da] = readEDF(fullfile(subls(ii).folder, subls(ii).name));
            da = single(da)';
        end
        if nn == 12 && ismember(ii, [11, 12, 13])
            ich = [5, 1; 1, 2; 2, 3; 3, 9; 5, 6; 6, 7; 7, 8; 8, 9; ...
                15, 16; 16, 17; 17, 18; 18, 19; 15, 21; 21, 22; ...
                22, 23; 23, 19; 11, 12; 12, 13];
            signal = prepSignal(da(:, ich(:, 1)) - da(:, ich(:, 2)), sfreq, dfs);
        else
            [~, ich] = ismember(chs, hdr.label(:));
            signal = prepSignal(da(:, ich), sfreq, dfs);
        end
        [segments, labels] = prepLabeledSegment(signal, dfs, subls(ii).evs, win, step, thr);
        isin = labels == 1 | (labels == 0 & mod((1:length(labels))', 30) == 9);
        subls(ii).X = segments(isin, :, :, :);
        subls(ii).Y = labels(isin);
    end

    segments = permute(cat(1, subls(:).X), [1, 4, 3, 2]);
    labels = cat(1, subls(:).Y);
    t1 = 1;
    t2 = find(diff(labels) == -1, 1, "first");
    writeNPY(segments(t1:t2, :, :, :), ...
        fullfile(out_folder, [fls(nn).name, '_X.npy']));
    writeNPY(labels(t1:t2), ...
        fullfile(out_folder, [fls(nn).name, '_Y.npy']));
end




