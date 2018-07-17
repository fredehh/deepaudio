direc=dir('spectro_h5');%Input path to spectrograms saved in H5
rng(1)
%%
freqs=128;
frames=zeros(length(direc)*2999,freqs);
for k=3:length(direc)
    temp=h5read(direc(k).name,'/SG');
    temp=temp(1:freqs,:);
    temp= temp./abs(max(max((temp))));
    for i=1:size(temp,2)
        frames((k-1)*2999+i,:)=temp(:,i);
    end
end%Create dataset of frames
%%
[coeff,score,latent]=pca(frames);%PCA
%%
CVpcs=zeros(30,5);
for p=1:30 
pcs=p  
p30= zeros(1,4) %p ;%Number of principal components
means=60 ;%Number of mean-frames projected onto principal components
features=means*pcs;
pcavec=zeros(1242,features);
for k=3:length(direc)%Load specs and create features
    temp=h5read(direc(k).name,'/SG');
    tempi=temp(1:freqs,:);   
    for i=1:means%features/pcs
        if i==1
            temp2=mean(tempi(:,1:50),2);
        elseif i==means%features/pcs
             temp2=mean(tempi(:,(i-1)*50+1:2999),2);
        else
            temp2=mean(tempi(:,(i-1)*50+1:i*50),2);
        end
        pcavec(k,i*pcs)=coeff(:,pcs)'*temp2; 
        for h=1:pcs-1
            pcavec(k,i*pcs-h)=coeff(:,h)'*temp2;
        end
    end
end
%load annotations
swap2train=0;
swap2val=4;
TrainAnnoId=[1,2,3];

    
for u=1:4 %Cross-validation
    if u==1
        swap2val=4;
        swap2train=4;
    else
        swap2val=TrainAnnoId(u-1) ;
        TrainAnnoId(u-1)=swap2train;
        swap2train=u-1;
    end

anno=csvread('annotations.csv');
testanno=[0,0,0,0];
valanno=[0,0,0,0];
trainanno=[0,0,0,0];
decision=rand(size(anno,1));
for k=1:size(anno,1)%Split into validation and training fold, and swap decisions/excerpts)
    if decision(k)<=0.5
       anno(k,4)=0;
       win=anno(k,2);
       lose=anno(k,3);
       anno(k,2)=lose;
       anno(k,3)=win;
    end
    if ismember(anno(k,1),TrainAnnoId)
        trainanno=[trainanno;anno(k,:)];
    end
    if anno(k,1)==swap2val
        valanno=[valanno;anno(k,:)]; 
    end
    if anno(k,1)==5
        testanno=[testanno;anno(k,:)];
    end
end

trainanno=trainanno(2:size(trainanno,1),:);
valanno=valanno(2:size(valanno,1),:);
testanno=testanno(2:size(testanno,1),:);

trainsize=size(trainanno,1);
trainspec=zeros(trainsize,features);
trainy=zeros(trainsize,1) ;

valsize=size(valanno,1);
valy=zeros(valsize,1);
valspec=zeros(valsize,features);

testsize=size(testanno,1);
testspec=zeros(testsize,features);
testy=zeros(testsize,1) ;

for k=1:trainsize  % Create training set of features and labels
    temp=pcavec(trainanno(k,2),:)-pcavec(trainanno(k,3),:);
    trainspec(k,:)=temp;
    trainy(k)=trainanno(k,4);
end
for k=1:valsize %Create validaiton set of features and labels
    temp=pcavec(valanno(k,2),:)-pcavec(valanno(k,3),:);
    valspec(k,:)=temp;
    valy(k)=valanno(k,4);
end

for k=1:testsize%Create test set of features and labels
    temp=pcavec(testanno(k,2),:)-pcavec(testanno(k,3),:);
    testspec(k,:)=temp;  
    testy(k)= testanno(k,4);
end


params.distrib = 'binom';
params.link = 'logit';
[what,params] = glmfitting(trainspec,trainy,params);
muhat = glmpredict(valspec,what,params);


sqerr=mean((muhat-valy).^2);

MisClassRate=sum(round(abs(muhat-valy)));
MisClassRate=MisClassRate/size(muhat,1);
CVpcs(p,u)=MisClassRate;
 %p30(1,u)=MisClassRate;
end
p
end
%%