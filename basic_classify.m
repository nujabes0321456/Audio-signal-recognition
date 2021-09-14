clear;clc;
%% Read Files
folder=fullfile(pwd,'data_im')  %Ū����Ʀs���Ƨ��W�١A'pwd'���ثe��Ƨ����W��
imgs=imageDatastore(folder,...  %Ū����Ƨ������Ϥ�
    "IncludeSubfolders",true,...
    "LabelSource","foldernames",...
    "FileExtensions",{'.jpeg','.png','.jpg','.bmp'})

N=length(imgs.Files)
idx=randperm(N); %�إߥ��ö��Ǫ�1*N�x�}idx
for i=1:16
    subplot(4,4,i)  %�@�����16�i�Ӥ�
    img=imread(imgs.Files{idx(i)}); %Ū�����N�Ӥ�
    [m,n,k]=size(img);  % ����Ӥ��j�p => ���X[�� ,�e, �C��]
    imshow(img) %��ܷӤ�
end

labelCount = countEachLabel(imgs) %��ܦU���O�Ӥ��`��

%% Data Pre-Processing
numTrainFiles=0.8; % 80%����Ʒ�@�V�m��; 20%��@���ն�
[Train,Test] = splitEachLabel(imgs,numTrainFiles,'randomize');
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-5,5], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5])
imageSize = [m,n,k];
for i=1:length(Train.Files)
    x(:,:,:,i)=imread(Train.Files{i});  %Ū��Train�Ӥ��w
    xl{i}=cellstr(Train.Labels(i));  %Ū��Train��ƪ�Label���নString
end
for i=1:length(Test.Files)
    x2(:,:,:,i)=imread(Test.Files{i});  %Ū��Test�Ӥ��w
end
disp('read ok')
augimds = augmentedImageDatastore(imageSize,x,Train.Labels,'DataAugmentation',imageAugmenter);
Validimds = augmentedImageDatastore(imageSize,x2,Test.Labels)
labelTrainCount = countEachLabel(Train) %���Training���O�Ӥ��`��
labelTestCount = countEachLabel(Test) %���Test���O�Ӥ��`��

%% Network Setting

Numoutclass=3; %�X��class(����Ʈw��'a','b','c'�T��)
layers = [
    imageInputLayer([m n k],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([5 5],"Name","maxpool","Padding","same")
    convolution2dLayer([3 3],16,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","batchnorm2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(100,"Name","fc1")
    fullyConnectedLayer(Numoutclass,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
plot(layerGraph(layers));

%% Training Option
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0005, ...
    'MaxEpochs',200, ...    % �̰��^�N����
    'MiniBatchSize',5,...   %�@���B��ɭn�B��h�ָ��
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment',"gpu",...    %�ϥ�cpu/gpu�i��B��A�n�ϥ�gpu�ݥt�~�]�w
    'ValidationData',Validimds, ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Training
net = trainNetwork(Validimds,layers,options);
YPred=classify(net,Test); % �z�L�V�m�n�����g������Test Data�i������w��
accuracy = sum(YPred == Test.Labels)/numel(Test.Labels)

%% Confusion Matrix
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(Test.Labels,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(2,pwd,'jpeg'); %%�x�sConfusion Matrix���Ϲ�

save networkfile.mat
disp('Model saved')
