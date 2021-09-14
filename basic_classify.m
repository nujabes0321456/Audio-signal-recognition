clear;clc;
%% Read Files
folder=fullfile(pwd,'data_im')  %讀取資料存放資料夾名稱，'pwd'為目前資料夾的名稱
imgs=imageDatastore(folder,...  %讀取資料夾中的圖片
    "IncludeSubfolders",true,...
    "LabelSource","foldernames",...
    "FileExtensions",{'.jpeg','.png','.jpg','.bmp'})

N=length(imgs.Files)
idx=randperm(N); %建立打亂順序的1*N矩陣idx
for i=1:16
    subplot(4,4,i)  %一次顯示16張照片
    img=imread(imgs.Files{idx(i)}); %讀取任意照片
    [m,n,k]=size(img);  % 抓取照片大小 => 給出[長 ,寬, 顏色]
    imshow(img) %顯示照片
end

labelCount = countEachLabel(imgs) %顯示各類別照片總數

%% Data Pre-Processing
numTrainFiles=0.8; % 80%的資料當作訓練集; 20%當作測試集
[Train,Test] = splitEachLabel(imgs,numTrainFiles,'randomize');
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-5,5], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5])
imageSize = [m,n,k];
for i=1:length(Train.Files)
    x(:,:,:,i)=imread(Train.Files{i});  %讀取Train照片庫
    xl{i}=cellstr(Train.Labels(i));  %讀取Train資料的Label並轉成String
end
for i=1:length(Test.Files)
    x2(:,:,:,i)=imread(Test.Files{i});  %讀取Test照片庫
end
disp('read ok')
augimds = augmentedImageDatastore(imageSize,x,Train.Labels,'DataAugmentation',imageAugmenter);
Validimds = augmentedImageDatastore(imageSize,x2,Test.Labels)
labelTrainCount = countEachLabel(Train) %顯示Training類別照片總數
labelTestCount = countEachLabel(Test) %顯示Test類別照片總數

%% Network Setting

Numoutclass=3; %幾個class(此資料庫有'a','b','c'三類)
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
    'MaxEpochs',200, ...    % 最高跌代次數
    'MiniBatchSize',5,...   %一次運算時要運算多少資料
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment',"gpu",...    %使用cpu/gpu進行運算，要使用gpu需另外設定
    'ValidationData',Validimds, ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Training
net = trainNetwork(Validimds,layers,options);
YPred=classify(net,Test); % 透過訓練好的神經網路對Test Data進行分類預測
accuracy = sum(YPred == Test.Labels)/numel(Test.Labels)

%% Confusion Matrix
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(Test.Labels,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(2,pwd,'jpeg'); %%儲存Confusion Matrix的圖像

save networkfile.mat
disp('Model saved')
