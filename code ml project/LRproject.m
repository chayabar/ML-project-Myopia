fileName = "myopia.csv"; %name pf file with data
myopiaDS = dataset('xlsfile', fileName); %load to matlab
%gender=categorical(myopiaDS.gender)=='Female';
x = [myopiaDS.sporthr,myopiaDS.readhr,myopiaDS.comphr,myopiaDS.studyhr,myopiaDS.tvhr,double(categorical(myopiaDS.mommy)=='Yes'),double(categorical(myopiaDS.dadmy)=='Yes')]; %create the x with relevant data to research
y = categorical(categorical(myopiaDS.myopic)=='Yes'); %the classification
%---------------------------------------------------------------
%logistic regression
%myopiaDS.diopterhr, myopiaDS.studyhr,myopiaDS.tvhr,myopiaDS.readhr,
%,dummyvar(nominal(gender)), ,,myopiaDS.age,,...

%x = [myopiaDS.comphr ,myopiaDS.diopterhr  ,myopiaDS.sporthr,double(categorical(myopiaDS.mommy)=='Yes'),double(categorical(myopiaDS.dadmy)=='Yes')]; %create the x with relevant data to research
pos=find(y=='true'); %find the indexs of positive labels
neg=find(y=='false'); %find the indexs of negative labels

x=[x(pos,:);x(neg(1:length(pos)),:)];
y=[y(pos,:);y(neg(1:length(pos)),:)];
numSamples = length(x); %number of samples

LRMistakes=[];
LRf1=[];

for i=1:50 %50 times to calculate the mean of variables TP, FN, FP,TN, for more accurate results
	randomVec = randperm(numSamples,numSamples); %random vector with numbers between 0-numSamples (single apearance)
    train70 = randomVec(1: round(numSamples*0.7));
    test30 = randomVec(round(numSamples*0.7):numSamples);
        
	b = mnrfit(x(train70,:),y(train70,:)); %build model (logistic regression) of 70% of data
	evaluation = mnrval(b,x(test30,:)); %evaluate the model on 30% of data (the data the model didnt see on train)
	predictedY = evaluation(:,1)<evaluation(: ,2); %turn the percent to 1/0
	%add to sum of the variables the results
    index_pos=find(y(test30,:)=='true'); %find the indexs of positive labels
    index_neg=find(y(test30,:)=='false'); %find the indexs of negative labels
    
%     TP = sum(predictedY(index_pos)==1); %find TP,FN,FP,TN
%     FN = sum(predictedY(index_pos)==0);
%     FP = sum(predictedY(index_neg)==1);
%     TN = sum(predictedY(index_neg)==0);
    
    TN = sum(predictedY(index_pos)==1); %find TP,FN,FP,TN
    FP = sum(predictedY(index_pos)==0);
    FN = sum(predictedY(index_neg)==1);
    TP = sum(predictedY(index_neg)==0);
    
    Precision=TP/(TP+FP); %caculate Precision, Recall,F1
    Recall=TP/(TP+FN);
    F1=2 * Precision * Recall / (Precision + Recall);
    LRf1=[LRf1 F1];
    
    num_mistakes=length(find(~(categorical(predictedY)==(y(test30,:)))));
        
    LRMistakes=[LRMistakes num_mistakes/length(test30)];
end
figure % new figure
ax1 = subplot(2,1,1); % top subplot
ax2 = subplot(2,1,2); % bottom subplot

plot(ax1, 1:50, LRMistakes)
title(ax1,'error rate')
plot(ax2, 1:50,LRf1 )
title(ax2,'F1')
sum(LRMistakes)/50
sum(LRf1)/50
