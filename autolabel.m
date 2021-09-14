clear;clc;
load networkfile.mat
folder=fullfile(pwd,'nor120s200160/')%�ثe��Ƨ����W��
imgs=imageDatastore(folder,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames",...
    "FileExtensions",{'.jpeg','.png','.jpg','.bmp'})
N=length(imgs.Files)
Newpath=fullfile(pwd,'auto');% ���w�s�и�Ƨ����W��
if (exist(Newpath)>0)
    rmdir(Newpath,"s")
end
mkdir(Newpath)%�s�и�Ƨ�
da=countEachLabel(imgs)
labels={'a';'b';'c'};
for i=1:length(labels)
    mkdir (fullfile(Newpath,labels{i}));
end
for i=1:N
    Originalimg=imread(imgs.Files{i});  %Ū���Ӥ�
    [m,n,k]=size(Originalimg);% ����Ӥ��j�p => ���X[�� ,�e, �C��]
    img=imcrop(Originalimg,centerCropWindow2d(size(Originalimg),[200 160]));%�Ũ��Ϥ��d��
    [filepath,name,ext] = fileparts( imgs.Files{i});
    % �Ϥ��j�p�ഫ [��l��,��l�e]=>[���w��, ���w�e]
    Resizeimg=imresize(img,[250,312]);
    
    YPred(i)=classify(net,Resizeimg);
    imwrite(Originalimg,...  %�s�ɮצܫ��w��Ƨ�
        [Newpath,'/',char(YPred(i)),'/',[name,'.png']]);
    if(mod(i,500)==0)
        disp(['progress:',num2str(i)]);
    end
end
predictclass=categories(YPred)
disp('OK');

