function Prepare_TrainData_HR_LR_Blur()
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
kernelsize = 25;
deviation = 1.6;
sigma = 2;
for IdxIm = 1:nb_im
    fprintf('IdxIm=%d\n', IdxIm);
    ImHR = DIV2K_HR{IdxIm};
    ImLR = imresize_Blur(ImHR, kernelsize, deviation, sigma);
    % name image
    digit = IdxIm;
    fileName = num2str(IdxIm);
    %while digit < 100000  % for Flickr2K
    while digit < 1000
        fileName = ['0', fileName];
        digit = digit*10;
    end

%     FolderLRx3 = fullfile(path_save, 'DIV2K_train_LR_Blur', 'K25N2');
    FolderLRx3 = fullfile(path_save, [data_name,'_LR_Blur'], 'K25N2');
    
    if ~exist(FolderLRx3)
        mkdir(FolderLRx3)
    end

    NameLRx3 = fullfile(FolderLRx3, [fileName, 'k25n2.png']);
    % save image
    imwrite(ImLR, NameLRx3, 'png');
end


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