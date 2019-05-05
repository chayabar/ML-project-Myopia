
fileName = "myopia.csv"; %name pf file with data
myopiaDS = dataset('xlsfile', fileName); %load to matlab
gender=categorical(myopiaDS.gender)=='Female';
%x = [myopiaDS.age,gender,myopiaDS.sporthr,myopiaDS.readhr,...
%    myopiaDS.comphr,myopiaDS.studyhr,myopiaDS.tvhr,myopiaDS.diopterhr,categorical(myopiaDS.mommy)=='Yes',categorical(myopiaDS.dadmy)=='Yes']; %create the x with relevant data to research
y = categorical(categorical(myopiaDS.myopic)=='Yes'); %the classification
x = [myopiaDS.sporthr,myopiaDS.readhr,myopiaDS.comphr,myopiaDS.studyhr,myopiaDS.tvhr,double(categorical(myopiaDS.mommy)=='Yes'),double(categorical(myopiaDS.dadmy)=='Yes')]; %create the x with relevant data to research
pos=find(y=='true'); %find the indexs of positive labels
neg=find(y=='false'); %find the indexs of negative labels
x=[x(pos,:);x(neg(1:length(pos)),:)];
y=[y(pos,:);y(neg(1:length(pos)),:)];
numSamples = length(x); %number of samples



%knn
KNNMistakes=[];
KNNclassError=[];
KNNf1=[];
for k=2:50
    num_mistakes_sum=0;
    classError_sum=0;
    F1_sum=0;
    for i=1:50
        randomVec = randperm(numSamples,numSamples); %random vector with numbers between 0-numSamples (single apearance)
        train70 = randomVec(1: round(numSamples*0.7));
        test30 = randomVec(round(numSamples*0.7):numSamples);
        
        KNNmdl=fitcknn(x(train70,:),y(train70),'NumNeighbors', k);
        predictedY=predict(KNNmdl,x(test30,:));
        num_mistakes=length(find(~(predictedY==y(test30))));
        num_mistakes_sum=num_mistakes_sum+num_mistakes;
        
        CVKNNMdl = crossval(KNNmdl);
        classError = kfoldLoss(CVKNNMdl);
        classError_sum=classError_sum+classError;
        
        index_pos=find(y(test30)=='true'); %find the indexs of positive labels
        index_neg=find(y(test30)=='false'); %find the indexs of negative labels
%         TP = sum(predictedY(index_pos)=='true'); %find TP,FN,FP,TN
%         FN = sum(predictedY(index_pos)=='false');
%         FP = sum(predictedY(index_neg)=='true');
%         TN = sum(predictedY(index_neg)=='false');
        
        TN = sum(predictedY(index_pos)=='true'); %find TP,FN,FP,TN
        FP = sum(predictedY(index_pos)=='false');
        FN = sum(predictedY(index_neg)=='true');
        TP = sum(predictedY(index_neg)=='false');
        
        Precision=TP/(TP+FP); %calculate precision and recall
        Recall=TP/(TP+FN);
        F1=2 * Precision * Recall / (Precision + Recall); %finally calculate F1
        if (isnan(F1))
            F1=0;
        end
        F1_sum=F1_sum+F1; %add to counter
    end
    
    KNNMistakes=[KNNMistakes (num_mistakes_sum/50)/length(test30)];
    KNNclassError=[KNNclassError classError_sum/50];
    KNNf1=[KNNf1 F1_sum/50];
end
figure % new figure
title('KNN')
ax1 = subplot(3,1,1); % top subplot
ax2 = subplot(3,1,2); % middle subplot
ax3 = subplot(3,1,3); % bottom subplot

plot(ax1, 2:50, KNNMistakes)
title(ax1,'error rate')
plot(ax2, 2:50,KNNclassError )
title(ax2,'CV')
plot(ax3, 2:50,KNNf1 )
title(ax3,'F1')

