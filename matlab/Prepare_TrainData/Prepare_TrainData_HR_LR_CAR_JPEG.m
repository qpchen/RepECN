% addpath(genpath('/home/jumpchan/master/develop/practice/SISR/EDSR-PyTorch/matlab/JPEG-Encoder-Decoder'))
function Prepare_TrainData_HR_LR_CAR_JPEG()
%% settings
% path_save = './DIV2K';
% path_src = './DIV2K/DIV2K_train_HR';
% path_save = './test';
% path_src = './test/HR';
data_name = 'Custom';
path_save = ['./', data_name];
path_src = ['./',data_name,'/',data_name,'_HR'];
ext               =  {'*.jpg','*.png','*.bmp'};
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(path_src, ext{i})));
end
nb_im = length(filepaths);
DIV2K_HR = [];

for idx_im = 1:nb_im
    fprintf('Read HR :%d\n', idx_im);
    ImHR = imread(fullfile(path_src, filepaths(idx_im).name));
    DIV2K_HR{idx_im} = ImHR;
end
%% generate and save LR via imresize() with Bicubic
quality_all = [10, 20, 30, 40]; % noise level
for quality = quality_all
    for IdxIm = 1:nb_im
        fprintf('IdxIm=%d\n', IdxIm);
        ImHR = DIV2K_HR{IdxIm};
        %[bitStr, imgDimensions] = jpegEncoder(ImHR, quality);
        % name image
        digit = IdxIm;
        fileName = num2str(IdxIm);
        %while digit < 100000  % for Flickr2K
        while digit < 1000
            fileName = ['0', fileName];
            digit = digit*10;
        end
    
%         FolderLR = fullfile(path_save, 'DIV2K_train_LR_JPEG', ['Q', num2str(quality)]);
        FolderLR = fullfile(path_save, [data_name,'_LR_JPEG'], ['Q', num2str(quality)]);
        
        if ~exist(FolderLR)
            mkdir(FolderLR)
        end
    
        NameLR = fullfile(FolderLR, [fileName, 'q', num2str(quality), '.jpg']);
        % save image
        imwrite(ImHR, NameLR, 'jpeg', 'Quality', quality);
    end
end

end
