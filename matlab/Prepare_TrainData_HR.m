function Prepare_TrainData_HR()
clear all; close all; clc
path_original = './OriginalTestData';
% dataset  = {'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'};
dataset  = {'Custom'};
% dataset = {'DIV2K', 'Flickr2K'};
%dataset  = {'720P', '1080P', '4K'};
ext = {'*.jpg', '*.png', '*.bmp', '*.tif'};


for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s ', idx_im, name_im);
        im_ori = imread(fullfile(path_original, dataset{idx_set}, name_im));
        if size(im_ori, 3) == 1
            im_ori = cat(3, im_ori, im_ori, im_ori);
        end
        folder_HR = fullfile('./Prepare_TrainData', dataset{idx_set}, [dataset{idx_set}, '_HR']);
        if ~exist(folder_HR)
            mkdir(folder_HR)
        end
        % fn
        fn_HR = fullfile('./Prepare_TrainData', dataset{idx_set}, [dataset{idx_set}, '_HR'], [name_im(1:end-4), '.png']);
        imwrite(im_ori, fn_HR, 'png');
        fprintf('\n');
    end
    fprintf('\n');
end
end
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end











