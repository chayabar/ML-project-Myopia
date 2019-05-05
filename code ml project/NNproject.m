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

NNf1=[];
NNprecision=[];
NNrecall=[];
NNc=[];
NeuronsOptimal=-1; %to hold num of neuron with optimal results
F1High=0; %to hold highest F1 results (of net with NeuronsOptimal neorons)
BTP=0;
BFP=0;
BFN=0;
BTN=0;
for numNeurons=5:2:30 %loop to find the optimal number of number neuron for the net
	net=patternnet(numNeurons); %create net with numNeurons
	net.trainParam.showWindow=0;
	F1_counter=0; %sum of F1
    precision_counter=0;
    recall_counter=0;
    c_counter=0;
	for i =1:50 %loop to calculate the mean of F1 for mire accurate results
		[net, tr]=train(net,x',dummyvar(categorical(y))' ); %train the net
		testX = x(tr.testInd,:); %extract the test indexs
		testT = dummyvar(categorical(y(tr.testInd,:)))'; %get the real labels of the test set, transform to specific format
		testY = net(testX'); %find the prediction using the neural network we trained
		[c,cm] = confusion(testT,testY); %cm is a confusion matrix
        
% 		TP=cm(1,1); %extract TP, FP, FN, TN from cm
% 		FP=cm(1,2);
% 		FN=cm(2,1);
% 		TN=cm(2,2);

        %change classes
        TP=cm(2,2); %extract TP, FP, FN, TN from cm
		FP=cm(2,1);
		FN=cm(1,2);
		TN=cm(1,1);
        
		Precision=TP/(TP+FP); %calculate Precision,Recall
		Recall=TP/(TP+FN);
		F1=2 * Precision * Recall / (Precision + Recall); %F1 CALCULATION
        if (isnan(F1))
            F1=0;
        end
        if (isnan(Precision))
            Precision=0;
        end
        if (isnan(Recall))
            Recall=0;
        end
		F1_counter=F1_counter+F1; %add to sum
        precision_counter=precision_counter+Precision;
        recall_counter=recall_counter+Recall;
        c_counter=c_counter+c;
	end
	F1Mean=F1_counter/50 %devide to get mean
    NNf1=[NNf1 F1Mean];
    NNprecision=[NNprecision precision_counter/50];
    NNrecall=[NNrecall recall_counter/50];
    NNc=[NNc c_counter/50];
	if F1Mean>F1High  %if better then the previous result
			F1High=F1Mean; %update the optimal number of neorons and F1 score
			NeuronsOptimal=numNeurons;
            BTP=TP;
            BFP=FP;
            BFN=FN;
            BTN=TN;
    end 
end

figure % new figure
ax1 = subplot(3,1,1); % top subplot
ax2 = subplot(3,1,2); % bottom subplot
ax3 = subplot(3,1,3); % bottom subplot

plot(ax1, 5:2:30,NNf1 )
title(ax1,'F1')

plot(ax2, 5:2:30,NNprecision,5:2:30,NNrecall );%,5:2:30,NNrecall
title(ax2,'precision and recall')

plot(ax3, 5:2:30, NNc)
title(ax3, 'c - error rate')