function convertGray()
clear all; close all; clc
path_original = './OriginalTestData';
%dataset  = {'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'};
dataset = {'CBSD68'};
%dataset  = {'720P', '1080P', '4K'};
ext = {'*.jpg', '*.png', '*.bmp'};

for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        im_ori = imread(fullfile(path_original, dataset{idx_set}, name_im));
        if size(im_ori, 3) == 3
            im_gray = rgb2gray(im_ori);
        end

        folder_Gray = fullfile(path_original, 'BSD68');
        if ~exist(folder_Gray)
            mkdir(folder_Gray)
        end
        fn_Gray = fullfile(path_original, 'BSD68', name_im);
        imwrite(im_gray, fn_Gray, 'png');
        fprintf('\n');
    end
    fprintf('\n');
end
end







