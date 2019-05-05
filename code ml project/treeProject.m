%ML ex5
%chaya barbolin 208493965
load hospital
%1
x=[hospital.Age(:), hospital.Weight(:)]; %the model will be based on age and weight
y=hospital.BloodPressure(:,1); %Y vector
rtree=fitrtree(x,y); %create the tree
view(rtree, 'mode', 'graph'); %show the tree to user
%2
[resuberror,cvloss]=evaluateTree(rtree); %evaluate the tree
disp([ 'training sample MSE: ', num2str(resuberror) ]); %disply results to user
disp([ 'cross validation sample MSE: ', num2str(cvloss) ]);
%3
for j=0:2
	Mdl=prune(rtree ,'Level', j); %prune the tree 2 levels
	%view(Mdl, 'mode', 'graph'); %show the tree after prone
	[resuberror,cvloss]=evaluateTree(Mdl); %evaluate the tree
	disp(['results after puninig ', num2str(j),' levels']);
	disp([ 'training sample MSE: ', num2str(resuberror) ]); %disply results to user
	disp([ 'cross validation sample MSE: ', num2str(cvloss) ]);
end
%4
Xaxis=[]; %hold indexs of x
YtrainAxis=[]; %hold indexs for y1 - results on train set
YcvAxis=[]; %hold indexs for y2 - results on cross validation set
for i=0: max(rtree.PruneList) %for diffrent prune level hold results
   Xaxis=[Xaxis, i]; %add to vec
   pruneTree=prune(rtree ,'Level', i); %prune i level
   [resuberror,cvloss]=evaluateTree(pruneTree); %get the evaluation
   YtrainAxis=[YtrainAxis, resuberror]; %add to proper vectors
   YcvAxis=[YcvAxis,cvloss];
end
plot(Xaxis, YtrainAxis,'-o',Xaxis,YcvAxis,'-o','MarkerIndices',1:length(Xaxis)); %display results to user
legend('y = train set','y = cv set') %add labels
xlabel('level of prune')
ylabel('MSE')

%the function evaluateTree get tree to evaluate (Mdl)
%returns resuberror= evaluation of the training errors of the tree
%cvloss= evaluation of the cross validation errors of the tree

function [resuberror,cvloss] = evaluateTree(Mdl)
%Evaluate the tree performances using the training set 
resuberror = resubLoss(Mdl);
%Evaluate the tree performances using the cross validation set
cvMdl = crossval(Mdl);
cvloss = kfoldLoss(cvMdl);
end

%output
%training sample MSE: 23.2618
%cross validation sample MSE: 73.3227
%results after puninig 0 levels
%training sample MSE: 23.2618
%cross validation sample MSE: 71.004
%results after puninig 1 levels
%training sample MSE: 23.3552
%cross validation sample MSE: 77.3424
%results after puninig 2 levels
%training sample MSE: 23.5484
%cross validation sample MSE: 80.5374

%clearly we can see that CV MSE is much higher 
%and we expect that because the model didnt see this set before,
%therefor the resultats will be less than the training set (reminder- low MSE is more accurate results)
%as we prune more the tree the results will be less accurate->less information