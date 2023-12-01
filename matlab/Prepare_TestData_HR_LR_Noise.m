function Prepare_TestData_HR_LR_Noise()
clear all; close all; clc
path_original = './OriginalTestData';
%dataset  = {'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'};
%dataset  = {'720P', '1080P', '4K'};
% dataset  = {'720P'};
ext = {'*.jpg', '*.png', '*.bmp', '*.tif'};

degradation = 'Gray_Noise'; % Noise, Blur, JPEG, Gray_Noise
% noise level
%sigma_all = [15, 25, 50]; % noise level
%sigma_all = [10, 30, 70]; % noise level
% Blur setting
kernelsize = 25;
deviation = 1.6;
%sigma = 2;
% CAR JPEG quality
quality_all = [10, 20, 30, 40]; % CAR JPEG quality
if strcmp(degradation, 'Blur') 
    sigma_all = 2;
    prefix = 'k25n';
    dataset  = {'McMaster', 'Kodak24', 'Urban100'};
elseif strcmp(degradation, 'Noise') 
    sigma_all = [10, 30, 70];
%     sigma_all = [15, 25, 50];
    prefix = 'n';
    dataset  = {'CBSD68', 'Kodak24', 'McMaster', 'Urban100'};
elseif strcmp(degradation, 'Gray_Noise') 
    sigma_all = [10, 30, 70];
%     sigma_all = [15, 25, 50];
    prefix = 'n';
    dataset  = {'Set12', 'BSD68', 'Urban100_Gray'};
elseif strcmp(degradation, 'JPEG') 
    sigma_all = 0;
    prefix = 'q';
    dataset  = {'LIVE1', 'Classic5'};
end

for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        im_HR = imread(fullfile(path_original, dataset{idx_set}, name_im));
        if ~strcmp(degradation, 'Gray_Noise') % not gray dataset
            if size(im_HR, 3) == 1
                im_HR = cat(3, im_HR, im_HR, im_HR);
            end
        end
        if ~strcmp(degradation, 'JPEG') % not JPEG dataset, which use sigma
            for sigma = sigma_all
                fprintf('S%d ', sigma);
                randn('seed',0); % For test data, fix seed. But, DON'T fix seed, when preparing training data.
                if strcmp(degradation, 'Noise')
                    im_LR = imresize_Noise(im_HR, sigma);
                elseif strcmp(degradation, 'Gray_Noise')
                    im_LR = imresize_Noise(im_HR, sigma); % sigma=1.6
                elseif strcmp(degradation, 'Blur')
                    im_LR = imresize_Blur(im_HR, kernelsize, deviation, sigma); % noise level sigma=30
                end
                % folder
                folder_HR = fullfile('./HR', dataset{idx_set}, [prefix, num2str(sigma)]);
                folder_LR = fullfile(['./LR/LR', degradation], dataset{idx_set}, [prefix, num2str(sigma)]);
                if ~exist(folder_HR,"dir")
                    mkdir(folder_HR)
                end
                if ~exist(folder_LR,"dir")
                    mkdir(folder_LR)
                end
                % fn
                fn_HR = fullfile('./HR', dataset{idx_set}, [prefix, num2str(sigma)], [name_im(1:end-4), '_HR_', prefix, num2str(sigma), '.png']);
                fn_LR = fullfile(['./LR/LR', degradation], dataset{idx_set}, [prefix, num2str(sigma)], [name_im(1:end-4), '_LR', degradation, '_', prefix, num2str(sigma), '.png']);
                imwrite(im_HR, fn_HR, 'png');
                imwrite(im_LR, fn_LR, 'png');
            end
        else
            for quality = quality_all
                fprintf('Q%d ', quality);
                % folder
                folder_HR = fullfile('./HR', dataset{idx_set}, [prefix, num2str(quality)]);
                folder_LR = fullfile(['./LR/LR', degradation], dataset{idx_set}, [prefix, num2str(quality)]);
                if ~exist(folder_HR,"dir")
                    mkdir(folder_HR)
                end
                if ~exist(folder_LR,"dir")
                    mkdir(folder_LR)
                end
                % fn
                fn_HR = fullfile('./HR', dataset{idx_set}, [prefix, num2str(quality)], [name_im(1:end-4), '_HR_', prefix, num2str(quality), '.png']);
                fn_LR = fullfile(['./LR/LR', degradation], dataset{idx_set}, [prefix, num2str(quality)], [name_im(1:end-4), '_LR', degradation, '_', prefix, num2str(quality), '.jpg']);
                imwrite(im_HR, fn_HR, 'png');
                imwrite(im_HR, fn_LR, 'jpeg', 'Quality', quality);
            end
        end
        fprintf('\n');
    end
    fprintf('\n');
end
end


function ImLR = imresize_Noise(ImHR, sigma)
% ImLR and ImHR are uint8 data
ImDown = single(ImHR); % 0-255
ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
ImLR = uint8(ImDownNoise); % 0-255
end

function ImLR = imresize_Blur(ImHR, kernelsize, deviation, sigma)
% ImLR and ImHR are uint8 data
% downsample by Bicubic
kernel  = fspecial('gaussian',kernelsize,deviation);
blur_HR = imfilter(ImHR,kernel,'replicate');
% ImLR = imresize(blur_HR, 1/scale, 'nearest');
ImDown = single(blur_HR); % 0-255
ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
ImLR = uint8(ImDownNoise); % 0-255
end






