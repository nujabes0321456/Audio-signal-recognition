clear;clc;
load networkfile.mat
folder=fullfile(pwd,'nor120s200160/')%目前資料夾的名稱
imgs=imageDatastore(folder,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames",...
    "FileExtensions",{'.jpeg','.png','.jpg','.bmp'})
N=length(imgs.Files)
Newpath=fullfile(pwd,'auto');% 給定新創資料夾的名稱
if (exist(Newpath)>0)
    rmdir(Newpath,"s")
end
mkdir(Newpath)%新創資料夾
da=countEachLabel(imgs)
labels={'a';'b';'c'};
for i=1:length(labels)
    mkdir (fullfile(Newpath,labels{i}));
end
for i=1:N
    Originalimg=imread(imgs.Files{i});  %讀取照片
    [m,n,k]=size(Originalimg);% 抓取照片大小 => 給出[長 ,寬, 顏色]
    img=imcrop(Originalimg,centerCropWindow2d(size(Originalimg),[200 160]));%剪取圖片範圍
    [filepath,name,ext] = fileparts( imgs.Files{i});
    % 圖片大小轉換 [原始長,原始寬]=>[指定長, 指定寬]
    Resizeimg=imresize(img,[250,312]);
    
    YPred(i)=classify(net,Resizeimg);
    imwrite(Originalimg,...  %存檔案至指定資料夾
        [Newpath,'/',char(YPred(i)),'/',[name,'.png']]);
    if(mod(i,500)==0)
        disp(['progress:',num2str(i)]);
    end
end
predictclass=categories(YPred)
disp('OK');

