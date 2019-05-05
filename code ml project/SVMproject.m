fileName = "myopia.csv"; %name pf file with data
myopiaDS = dataset('xlsfile', fileName); %load to matlab
%gender=categorical(myopiaDS.gender)=='Female';
x = [myopiaDS.sporthr,myopiaDS.readhr,myopiaDS.comphr,myopiaDS.studyhr,myopiaDS.tvhr,double(categorical(myopiaDS.mommy)=='Yes'),double(categorical(myopiaDS.dadmy)=='Yes')]; %create the x with relevant data to research
y = categorical(categorical(myopiaDS.myopic)=='Yes'); %the classification
%---------------------------------------------------------------

%x = [myopiaDS.comphr ,myopiaDS.diopterhr  ,myopiaDS.sporthr,double(categorical(myopiaDS.mommy)=='Yes'),double(categorical(myopiaDS.dadmy)=='Yes')]; %create the x with relevant data to research
pos=find(y=='true'); %find the indexs of positive labels
neg=find(y=='false'); %find the indexs of negative labels
x=[x(pos,:);x(neg(1:length(pos)),:)];
y=[y(pos,:);y(neg(1:length(pos)),:)];
numSamples = length(x); %number of samples

KernelFunc={'linear', 'rbf', 'gaussian'}; %dict to hold three type of Kernel Functions
for i = 1:3 %loop for each SVM model withd diffrent Kernel Functions and its calculations
	mis_counter=0; %mismatch counter
	F1_counter=0; %F1 counter
    precision_counter=0;
    recall_counter=0;
    for j=1:50
		randomVec = randperm(numSamples,numSamples); %random vector with numbers between 0-numSamples (single apearance)
        train70 = randomVec(1: round(numSamples*0.7));
        test30 = randomVec(round(numSamples*0.7):numSamples);
        SVMModel=fitcsvm(x(train70,:), y(train70),'KernelFunction',KernelFunc{i}); %train SVM model using train set and their labels, specify the KernelFunction we want
        [predictedY,score] = predict(SVMModel,x(test30,:)); %predict the labels of test set using the Model
        
        index_pos=find(y(test30,:)=='true'); %find the indexs of positive labels
        index_neg=find(y(test30,:)=='false'); %find the indexs of negative labels
        num_mistakes=length(find(~(categorical(predictedY)==(y(test30,:)))));
		mis_counter=mis_counter+num_mistakes; %add to counter
		predictedY=categorical(predictedY); 
        
% 		TP = sum(predictedY(index_pos)=='true'); %find TP,FN,FP,TN
% 		FN = sum(predictedY(index_pos)=='false');
% 		FP = sum(predictedY(index_neg)=='true');
% 		TN = sum(predictedY(index_neg)=='false');
        
      TN = sum(predictedY(index_pos)=='true'); %find TP,FN,FP,TN
		FP = sum(predictedY(index_pos)=='false');
		FN = sum(predictedY(index_neg)=='true');
		TP = sum(predictedY(index_neg)=='false');
        
		Precision=TP/(TP+FP); %calculate precision and recall
		Recall=TP/(TP+FN);
		F1=2 * Precision * Recall / (Precision + Recall); %finally calculate F1
        
        F1_counter=F1_counter+F1; %F1 counter
        precision_counter=precision_counter+Precision;
        recall_counter=recall_counter +Recall;
        
    end
    mistakes=mis_counter/50; %devide to get the mean
    disp([KernelFunc{i},' model results :']); %display the results to user
    disp([ num2str(mistakes),' mistakes in prediction on training set']);
    disp([ 'error rate=',num2str(mistakes/length(test30))]);
    disp([ 'F1=', num2str(F1_counter/50)]);
    disp([ 'Precision=', num2str(precision_counter/50)]);
    disp([ 'recall=', num2str(recall_counter/50)]);
    disp([ '--------------------------------']);
end